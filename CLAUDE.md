# CLAUDE.md — Label Studio ML Backend

AI assistant guide for understanding and developing in this repository.

---

## Repository Overview

**Label Studio ML Backend** is an SDK + web server framework that wraps ML model code and exposes it as a REST API that [Label Studio](https://labelstud.io/) can connect to for automated annotation, interactive labeling, and model training.

- **Package name**: `label-studio-ml`
- **Version**: `2.0.1dev0` (or `$VERSION_OVERRIDE`)
- **Entry point CLI**: `label-studio-ml` → `label_studio_ml/server.py:main`
- **Default port**: `9090`

---

## Directory Structure

```
label-studio-ml-backend/
├── label_studio_ml/            # Core SDK package
│   ├── __init__.py             # Package name + version
│   ├── model.py                # LabelStudioMLBase (base class for all backends)
│   ├── api.py                  # Flask REST API endpoints
│   ├── server.py               # CLI: init / start / deploy subcommands
│   ├── wsgi.py                 # WSGI entry point with logging setup
│   ├── cache.py                # SqliteCache for persistent project-scoped KV store
│   ├── response.py             # ModelResponse pydantic model
│   ├── exceptions.py           # AnswerException + exception_handler decorator
│   ├── utils.py                # Helper functions (label matching, image utils, etc.)
│   ├── ls_io.py                # Label Studio dataset download utility
│   ├── default_configs/        # Templates for `label-studio-ml create`
│   └── examples/               # 20+ ready-to-use ML backend implementations
│       ├── bert_classifier/
│       ├── easyocr/
│       ├── flair/
│       ├── gliner/
│       ├── grounding_dino/
│       ├── grounding_sam/
│       ├── huggingface_llm/
│       ├── huggingface_ner/
│       ├── interactive_substring_matching/
│       ├── langchain_search_agent/
│       ├── llm_interactive/
│       ├── mmdetection-3/
│       ├── nemo_asr/
│       ├── segment_anything_model/     # Original SAM (MobileSAM / SAM / ONNX)
│       ├── segment_anything_2_image/   # SAM 2 for images
│       ├── segment_anything_2_video/   # SAM 2 for video
│       ├── sklearn_text_classifier/
│       ├── spacy/
│       ├── tesseract/
│       ├── timeseries_segmenter/
│       ├── watsonx_llm/
│       └── yolo/
├── tests/                      # Core SDK test suite
├── requirements.txt            # Core dependencies
├── setup.py                    # Package setup
├── Makefile                    # install / test targets
├── codecov.yml                 # Coverage config
└── .github/workflows/          # CI/CD pipelines
```

Each example follows the same structure:
```
examples/<name>/
├── model.py            # ML implementation (inherits LabelStudioMLBase)
├── _wsgi.py            # WSGI entry point for gunicorn
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-base.txt
├── requirements-test.txt
├── test_api.py         # pytest tests
└── README.md
```

---

## Core Architecture

### Base Class: `LabelStudioMLBase`

All ML backends inherit from `LabelStudioMLBase` (`label_studio_ml/model.py`).

Key methods to override:

```python
class MyModel(LabelStudioMLBase):
    def setup(self):
        """Called on init. Set model_version, load models, etc."""
        self.set("model_version", "1.0.0")

    def predict(self, tasks, context=None, **kwargs):
        """
        Args:
            tasks: list of Label Studio task dicts
            context: dict with interactive labeling context (keypoints, boxes, etc.)
        Returns:
            list of prediction dicts OR ModelResponse
        """
        ...

    def fit(self, event, data, **kwargs):
        """
        Called by webhooks on annotation events.
        event: 'ANNOTATION_CREATED' | 'ANNOTATION_UPDATED' | 'ANNOTATION_DELETED' | 'START_TRAINING'
        """
        ...
```

Key properties and methods:

| Member | Description |
|---|---|
| `self.label_config` | Raw XML label config string |
| `self.parsed_label_config` | Parsed label config as dict |
| `self.label_interface` | `LabelInterface` object from SDK |
| `self.model_version` | Current semver model version |
| `self.set(key, value)` | Persist a string value to SQLite cache (project-scoped) |
| `self.get(key)` | Retrieve a cached value |
| `self.has(key)` | Check if a key exists in cache |
| `self.get_local_path(url, task_id=...)` | Download/resolve a URL to a local path |
| `self.preload_task_data(task)` | Recursively resolve URLs in task data |
| `self.get_first_tag_occurence(ctrl, obj)` | Find first matching tag pair in label config |
| `self.build_label_map(tag, names)` | Map model labels to Label Studio label values |

### Decorator Pattern (Alternative)

Instead of subclassing, you can register functions:

```python
from label_studio_ml.model import predict_fn, update_fn

@predict_fn
def my_predict(tasks, context, helper, **kwargs):
    ...

@update_fn
def my_update(event, data, helper, **kwargs):
    ...
```

### Flask API Endpoints (`label_studio_ml/api.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run predictions on tasks |
| `/setup` | POST | Initialize model with label config |
| `/webhook` | POST | Handle annotation events (training) |
| `/health`, `/` | GET | Health check |
| `/metrics` | GET | Metrics (returns empty `{}`) |

Authentication: HTTP Basic Auth via `BASIC_AUTH_USER` / `BASIC_AUTH_PASS` env vars (HMAC-safe comparison).

### Cache (`label_studio_ml/cache.py`)

- **Type**: `SqliteCache` (default), keyed by `(project_id, key)`
- **Storage**: `$MODEL_DIR/cache.db` (default: `./cache.db`)
- **Thread-safe**: uses `threading.Lock`
- `CACHE_TYPE` env var selects cache backend (only `sqlite` supported)

### Response Format (`label_studio_ml/response.py`)

```python
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

return ModelResponse(predictions=[
    PredictionValue(result=[...], score=0.9)
])
```

Or return a raw list of dicts (legacy format).

---

## Development Workflow

### Installation

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
```

### Create a New Backend

```bash
label-studio-ml create my_backend
cd my_backend
# Edit model.py to implement predict() and optionally fit()
label-studio-ml start my_backend
```

### Run an Existing Example

```bash
cd label_studio_ml/examples/<example_name>
docker-compose up
```

The backend starts at `http://localhost:9090`.

### Run Without Docker

```bash
label-studio-ml start label_studio_ml/examples/<example_name> -p 9090
```

### Testing

```bash
# Run all core tests
make test
# or
pytest tests/

# Run a specific example's tests
cd label_studio_ml/examples/<example_name>
pytest test_api.py
```

Test dependencies: `pytest>=6.2.5`, `pytest-cov>=3.0.0` (see `tests/requirements-test.txt`).

---

## Environment Variables

### Core Variables

| Variable | Default | Description |
|---|---|---|
| `VERSION_OVERRIDE` | `2.0.1dev0` | Override package version |
| `CACHE_TYPE` | `sqlite` | Cache backend type |
| `MODEL_DIR` | `.` | Directory for `cache.db` storage |
| `LOG_LEVEL` | inherited | Python logging level |
| `LABEL_STUDIO_URL` / `LABEL_STUDIO_HOST` | — | Label Studio instance URL |
| `LABEL_STUDIO_API_KEY` / `LABEL_STUDIO_ACCESS_TOKEN` | — | Legacy API token |
| `BASIC_AUTH_USER` | — | HTTP Basic Auth username |
| `BASIC_AUTH_PASS` | — | HTTP Basic Auth password |
| `WORKERS` | `1` | gunicorn worker count |
| `THREADS` | `8` | gunicorn thread count |
| `PORT` | `9090` | Server port |

> **Warning**: Only Legacy Tokens are supported for `LABEL_STUDIO_API_KEY`. Personal Tokens cause `401 Unauthorized`.

### Example-Specific Variables

| Example | Variable | Description |
|---|---|---|
| `segment_anything_model` | `SAM_CHOICE` | `MobileSAM` (default), `SAM`, or `ONNX` |
| `segment_anything_model` | `VITH_CHECKPOINT` | Path to `sam_vit_h_4b8939.pth` |
| `segment_anything_model` | `MOBILESAM_CHECKPOINT` | Path to `mobile_sam.pt` |
| `segment_anything_model` | `ONNX_CHECKPOINT` | Path to ONNX model |
| `segment_anything_2_image` | `DEVICE` | `cuda` or `cpu` |
| `segment_anything_2_image` | `MODEL_CONFIG` | SAM2 config YAML path |
| `segment_anything_2_image` | `MODEL_CHECKPOINT` | SAM2 checkpoint filename |
| `llm_interactive`, `langchain_search_agent` | `OPENAI_API_KEY` | OpenAI API key |
| `langchain_search_agent` | `GOOGLE_CSE_ID`, `GOOGLE_API_KEY` | Google Search credentials |

---

## CI/CD Pipelines (`.github/workflows/`)

| Workflow | Trigger | Description |
|---|---|---|
| `tests.yml` | push/PR/schedule | Matrix tests per example, coverage via Codecov |
| `build.yml` | push to master | Build and push Docker images to DockerHub |
| `build-pypi.yml` | release | Publish to PyPI |
| `release-pipeline.yml` | manual | Full release automation |
| `gitleaks.yml` | push/PR | Secret scanning |
| `pr-labeler.yml` | PR | Auto-label PRs |

Tests use a matrix from `docker-build-config.yml` — one job per ML backend. Heavy models (YOLO, NeMo, SAM) use `ubuntu-latest-4c-16gb` runners.

---

## Code Conventions

### Python Style

- PEP 8
- Type hints throughout (`Optional`, `List`, `Dict`, `Union`, `Tuple`)
- `logging` standard library for all logging; no `print()` in production code
- Cache values must be **strings** (`SqliteCache` enforces this)
- `self.set(key, json.dumps(obj))` / `json.loads(self.get(key))` for complex objects

### Model Class Patterns

```python
# Always call super().__init__() implicitly — do NOT override __init__ unless necessary
# Use setup() for initialization instead

class MyModel(LabelStudioMLBase):
    def setup(self):
        self.set("model_version", "1.0.0")
        # load model weights here

    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, value = self.get_first_tag_occurence('RectangleLabels', 'Image')
        label_map = self.build_label_map(from_name, model_class_names)
        # ... return predictions
```

### Label Mapping

Use `build_label_map()` to map model output labels → Label Studio label values. Supports `predicted_values` attribute in `<Label>` tags for custom mapping, falls back to case-insensitive name matching.

### Prediction Response Format

```python
# New style (preferred)
return ModelResponse(predictions=[{
    'result': [{
        'from_name': from_name,
        'to_name': to_name,
        'type': 'rectanglelabels',
        'value': {'x': 10, 'y': 20, 'width': 30, 'height': 40, 'rectanglelabels': ['Cat']},
        'score': 0.95
    }],
    'score': 0.95
}])

# Legacy style (also accepted)
return [{'result': [...], 'score': 0.95}]
```

---

## Docker Patterns

Each example's `docker-compose.yml` follows this pattern:

```yaml
version: "3.8"
services:
  ml-backend:
    build: .
    environment:
      - LABEL_STUDIO_URL=http://<host>:8080   # not localhost
      - LABEL_STUDIO_API_KEY=<legacy_token>
      - MODEL_DIR=/data/models
      - WORKERS=1
      - THREADS=8
    ports:
      - "9090:9090"
    volumes:
      - "./data/server:/data"
```

**Important**: Inside Docker, never use `localhost` for `LABEL_STUDIO_URL`. Use the host machine's actual IP (find with `ifconfig` / `ipconfig`).

### Using Pre-downloaded Model Weights

For `segment_anything_model`:
```yaml
volumes:
  - "./data/server:/data"
  - "/path/to/your/models:/app/models"   # mount local weights directory
environment:
  - SAM_CHOICE=SAM                        # or MobileSAM
  - VITH_CHECKPOINT=/app/models/sam_vit_h_4b8939.pth
  # - MOBILESAM_CHECKPOINT=/app/models/mobile_sam.pt
```

For `segment_anything_2_image`:
```yaml
volumes:
  - "./data/server:/data"
  - "/path/to/your/checkpoints:/sam2/checkpoints"
environment:
  - MODEL_CHECKPOINT=sam2.1_hiera_large.pt   # filename only
  - MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
```

### GPU Support

Uncomment the `deploy.resources.reservations.devices` block:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

### Rebuild Without Cache (for dependency updates)

```bash
docker compose build --no-cache
```

### Windows Line Ending Fix

```bash
git config --global core.autocrlf false
# then re-clone the repository
```

---

## Testing Patterns

### Core SDK Tests (`tests/`)

```python
# tests/test_api.py — Flask test client pattern
import pytest
from label_studio_ml.api import init_app
from mymodel import MyModel

@pytest.fixture
def client():
    app = init_app(MyModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    response = client.get('/')
    assert response.status_code == 200
```

### Example Tests (`test_api.py`)

Each example has its own `test_api.py` with integration tests using the Flask test client. Run with:

```bash
cd label_studio_ml/examples/<name>
pytest test_api.py -v
```

### Conditional Tests

Some tests skip when specific env vars are not set:
```python
import pytest, os
@pytest.mark.skipif(not os.getenv('ML_BACKEND'), reason='ML_BACKEND not set')
def test_something():
    ...
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `exec /app/start.sh: No such file or directory` (Windows) | Set `git config --global core.autocrlf false` and re-clone |
| `Bad Gateway` / `Service Unavailable` | Don't send concurrent requests; dev mode only |
| Predictions not showing | Set `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` |
| `no such file or directory` in logs | ML backend can't reach Label Studio data; set env vars above |
| `Unauthorized Error` | You're using a Personal Token; switch to a Legacy Token |
| Model slow / OOM | Increase Docker memory limits or add GPU support |

---

## GCP Deployment

```bash
label-studio-ml deploy gcp <ml-backend-dir> \
  --from=<model.py> \
  --gcp-project-id <id> \
  --label-studio-host https://app.heartex.com \
  --label-studio-api-key <key>
```

---

## Key External Dependencies

- `label-studio-sdk` — Official SDK (installed from git main branch)
- `Flask > 2.3` — Web framework
- `semver ~3.0.2` — Model versioning
- `requests ~2.31` — HTTP client
- `colorama ~0.4` — Colored terminal output
- `pydantic` — Response validation (via SDK)
