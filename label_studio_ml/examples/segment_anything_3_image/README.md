<!--
---
title: SAM 3 with Images
type: guide
tier: all
order: 16
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM 3 with Label Studio for Image Annotation
categories:
    - Computer Vision
    - Image Annotation
    - Object Detection
    - Segment Anything Model
image: "/guide/ml_tutorials/sam3-images.png"
---
-->

# Using SAM 3 with Label Studio for Image Annotation

[Segment Anything 3 (SAM 3)](https://ai.meta.com/blog/segment-anything-model-3/) is Meta's latest
foundation model for promptable segmentation, released in November 2025. Building on SAM and SAM 2,
SAM 3 introduces **text-prompted concept segmentation** alongside traditional point and box prompts.

## Key features

| Feature | SAM 1 | SAM 2 | SAM 3 |
|---------|-------|-------|-------|
| Point prompts | Yes | Yes | Yes |
| Box prompts | Yes | Yes | Yes |
| Text prompts | No | No | **Yes** |
| Pre-annotation | No | No | **Yes** |
| Interactive mode | Yes | Yes | Yes |

SAM 3 can **automatically pre-annotate** images by using label names (e.g. "person", "car") as text prompts
to detect and segment all matching instances. It also supports interactive point/box refinement.

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

### Requirements

- GPU with CUDA 12.6 or higher
- Python 3.12+
- PyTorch 2.7+
- HuggingFace account with access to [facebook/sam3](https://huggingface.co/facebook/sam3)

### Model access

SAM 3 checkpoints require access approval on HuggingFace:

1. Visit [facebook/sam3 on HuggingFace](https://huggingface.co/facebook/sam3) and request access.
2. Generate an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens).
3. Authenticate: `huggingface-cli login`

## Labeling configuration

SAM 3 works in two modes:

- **Pre-annotation mode** (no user interaction): The model uses label names from `BrushLabels`
  as text prompts to segment all matching objects.
- **Interactive mode**: The user places keypoints or draws rectangles, and SAM 3 returns
  segmentation masks.

All three control tags should be present in your labeling configuration:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="person" background="#FF0000"/>
    <Label value="car" background="#0d14d3"/>
  </BrushLabels>
  <KeyPointLabels name="tag2" toName="image" smart="true">
    <Label value="person" background="#000000" showInline="true"/>
    <Label value="car" background="#000000" showInline="true"/>
  </KeyPointLabels>
  <RectangleLabels name="tag3" toName="image" smart="true">
    <Label value="person" background="#000000" showInline="true"/>
    <Label value="car" background="#000000" showInline="true"/>
  </RectangleLabels>
</View>
```

> **Tip**: Choose label names that describe the visual concept you want to segment
> (e.g. "red car", "person with backpack"). SAM 3 understands natural language.

## Running with Docker

1. Start the ML backend on `http://localhost:9090`:

```bash
cd label_studio_ml/examples/segment_anything_3_image
docker-compose up
```

2. Validate that backend is running:

```bash
curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio: go to your project
   `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as the URL.

## Running from source

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
cd label_studio_ml/examples/segment_anything_3_image
pip install -r requirements.txt
```

2. Install SAM 3:

```bash
pip install git+https://github.com/facebookresearch/sam3.git
```

3. Authenticate with HuggingFace (for checkpoint download):

```bash
huggingface-cli login
```

4. Start the ML backend:

```bash
label-studio-ml start label_studio_ml/examples/segment_anything_3_image
```

5. Connect to the backend from Label Studio: go to your project
   `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as the URL.

## Configuration

Parameters can be set in `docker-compose.yml` or as environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | Device for inference (`cuda` required) |
| `SAM3_CHECKPOINT` | (auto-download) | Path to local `sam3.pt` checkpoint |
| `HF_TOKEN` | — | HuggingFace token for automatic checkpoint download |
| `BASIC_AUTH_USER` | — | Basic auth user for the model server |
| `BASIC_AUTH_PASS` | — | Basic auth password for the model server |
| `LOG_LEVEL` | `DEBUG` | Log level |
| `WORKERS` | `1` | Number of gunicorn workers |
| `THREADS` | `4` | Number of gunicorn threads |
| `LABEL_STUDIO_URL` | — | Label Studio instance URL |
| `LABEL_STUDIO_API_KEY` | — | Label Studio API key (Legacy Token only) |

### Using a pre-downloaded checkpoint

To avoid downloading the checkpoint at runtime, mount it as a volume:

```yaml
volumes:
  - "./data/server:/data"
  - "/path/to/your/checkpoints:/app/checkpoints"
environment:
  - SAM3_CHECKPOINT=/app/checkpoints/sam3.pt
```

## How it works

### Pre-annotation (text prompts)

When Label Studio calls `/predict` without interactive context, SAM 3 uses the label names
from your `BrushLabels` as text prompts. For example, if your labels are "person" and "car",
SAM 3 will find and segment all persons and cars in the image.

### Interactive segmentation (point / box prompts)

When a user places a keypoint or draws a rectangle in Label Studio's interactive mode,
the context is sent to SAM 3 which returns a precise segmentation mask for the indicated object.

- **Keypoints**: Click to add positive (foreground) or negative (background) points.
- **Rectangles**: Draw a bounding box around the target object.
