# Label Studio ML Backend 开发指南

## 一、整体架构

ML Backend 本质是一个 **Flask Web 服务**，Label Studio 通过 HTTP 调用它。框架已经封装好了所有接口，你只需要写一个继承 `LabelStudioMLBase` 的类，实现业务逻辑即可。

```
Label Studio ──POST /predict──▶ Flask (label_studio_ml/api.py)
                                       │
                                       ▼
                              你的 model.py (继承 LabelStudioMLBase)
                                       │
                                       ▼
                              你的模型推理代码
```

框架自动处理的部分：
- HTTP 路由（`/predict`、`/health`、`/webhook` 等）
- 认证（Basic Auth）
- 标注配置解析
- 模型版本管理
- 缓存（SQLite）

---

## 二、必须实现的接口

### `predict(tasks, context, **kwargs)` — 唯一必须实现的方法

```python
def predict(
    self,
    tasks: List[Dict],
    context: Optional[Dict] = None,
    **kwargs
) -> ModelResponse:
    ...
```

| 参数 | 说明 |
|---|---|
| `tasks` | Label Studio 传来的任务列表 |
| `context` | 交互式标注时的用户操作（点、框），非交互时为 `None` |
| 返回值 | `ModelResponse` 对象或 dict 列表 |

### 可选实现的方法

| 方法 | 触发时机 | 用途 |
|---|---|---|
| `setup()` | 服务启动时 | 加载模型权重、设置版本号 |
| `fit(event, data)` | 用户完成标注后 | 增量训练、主动学习 |

---

## 三、完整的 `model.py` 模板

```python
import os
import logging
from typing import List, Dict, Optional
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv('MODEL_PATH', './weights/model.pth')

# 在模块级别加载模型（只加载一次，不要放在 predict 里）
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f'Loading model from {MODEL_PATH}')
        # _model = YourModel.load(MODEL_PATH)
        logger.info('Model loaded')
    return _model


class MyMLBackend(LabelStudioMLBase):

    def setup(self):
        """服务启动时调用，设置版本号、提前加载模型"""
        self.set('model_version', 'v1.0.0')
        get_model()

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs
    ) -> ModelResponse:
        """
        核心预测方法。

        tasks 结构：
        [{
            'id': 1,
            'data': {'image': '/data/upload/1/xxx.jpg'},
            'annotations': [],
            ...
        }]

        context 结构（交互式标注时）：
        {
            'result': [{
                'type': 'keypointlabels',
                'value': {'x': 50.0, 'y': 50.0, 'keypointlabels': ['Cat']},
                'original_width': 1920,
                'original_height': 1080,
            }]
        }
        """
        # 解析 label config，获取控件信息
        from_name, to_name, value = self.get_first_tag_occurence('RectangleLabels', 'Image')

        predictions = []
        for task in tasks:
            # 1. 获取图片本地路径（自动处理本地/Docker/云存储）
            img_url = task['data'][value]
            local_path = self.get_local_path(img_url, task_id=task.get('id'))

            # 2. 加载图片
            image = Image.open(local_path).convert('RGB')
            image_width, image_height = image.size

            # 3. 模型推理
            model = get_model()
            # boxes, scores, labels = model.infer(image)

            # 4. 构造返回结果
            predictions.append({
                'result': [
                    {
                        'id': 'result_1',
                        'type': 'rectanglelabels',
                        'from_name': from_name,
                        'to_name': to_name,
                        'original_width': image_width,
                        'original_height': image_height,
                        'image_rotation': 0,
                        'value': {
                            'x': 10.0,      # 左上角 x，相对图片宽度的百分比 (0-100)
                            'y': 10.0,      # 左上角 y，相对图片高度的百分比 (0-100)
                            'width': 20.0,  # 宽度百分比
                            'height': 30.0, # 高度百分比
                            'rotation': 0,
                            'rectanglelabels': ['Cat'],
                        },
                        'score': 0.95,
                    }
                ],
                'score': 0.95,
            })

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        可选：用户完成标注后触发。

        event 取值：
          'ANNOTATION_CREATED'
          'ANNOTATION_UPDATED'
          'ANNOTATION_DELETED'
          'START_TRAINING'
        """
        logger.info(f'Received training event: {event}')
        # 增量训练逻辑...
```

---

## 四、返回格式详解

### 不同标注类型的 `value` 字段

**矩形框 (RectangleLabels)：**
```python
'type': 'rectanglelabels',
'value': {
    'x': 10.0,       # 百分比，相对图片宽度
    'y': 20.0,       # 百分比，相对图片高度
    'width': 30.0,
    'height': 40.0,
    'rotation': 0,
    'rectanglelabels': ['Cat'],
}
```

**关键点 (KeyPointLabels)：**
```python
'type': 'keypointlabels',
'value': {
    'x': 50.0,
    'y': 50.0,
    'width': 0.5,    # 点的显示大小
    'keypointlabels': ['Person'],
}
```

**Brush Mask (BrushLabels)：**
```python
from label_studio_converter import brush

mask = ...  # np.ndarray, shape (H, W), dtype uint8, 值为 0 或 255
rle = brush.mask2rle(mask)

'type': 'brushlabels',
'value': {
    'format': 'rle',
    'rle': rle,
    'brushlabels': ['Foreground'],
}
```

**分类 (Choices)：**
```python
'type': 'choices',
'value': {
    'choices': ['positive'],
}
```

**完整 result 条目结构：**
```python
{
    'id': 'unique_id',          # 唯一 ID，str
    'type': 'rectanglelabels', # 标注类型
    'from_name': from_name,    # label config 中控件的 name
    'to_name': to_name,        # label config 中数据对象的 name
    'original_width': 1920,    # 图片原始宽度（像素）
    'original_height': 1080,   # 图片原始高度（像素）
    'image_rotation': 0,
    'value': { ... },          # 上方各类型的 value
    'score': 0.95,             # 置信度（可选）
    'readonly': False,
}
```

---

## 五、`_wsgi.py` — 启动入口

```python
import os
import logging
from label_studio_ml.api import init_app
from model import MyMLBackend

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))

if __name__ == '__main__':
    # python _wsgi.py（开发调试用）
    app = init_app(model_class=MyMLBackend)
    app.run(host='0.0.0.0', port=9090, debug=False)
else:
    # gunicorn 启动（生产 / Docker 用）
    app = init_app(model_class=MyMLBackend)
```

**gunicorn 启动命令：**
```bash
gunicorn --bind 0.0.0.0:9090 --workers 1 --threads 8 '_wsgi:app'
```

> `WORKERS` 建议设为 1，模型推理通常不适合多进程（GPU 显存 / 模型状态共享问题）。

---

## 六、`get_local_path` — 获取图片本地路径

框架最重要的工具方法，自动处理各种图片来源：

```python
local_path = self.get_local_path(
    url,                       # task['data']['image']
    task_id=task.get('id')    # 必填，云存储 URI 解析需要
)
```

| 运行方式 | 图片来源 | 处理方式 |
|---|---|---|
| 本地 Python | Label Studio 同机 | 直接读文件系统，无需网络 |
| Docker 容器 | 宿主机 Label Studio | HTTP 下载，需要 `LABEL_STUDIO_API_KEY` |
| 任意 | S3 / GCS / Azure | 通过 Label Studio 代理下载 |

---

## 七、`label_config` 解析工具

```python
# 获取第一个匹配的控件对
from_name, to_name, value = self.get_first_tag_occurence('RectangleLabels', 'Image')
# from_name: 控件的 name 属性，如 "label"
# to_name:   数据对象的 name 属性，如 "image"
# value:     数据字段名，如 "image"（对应 task['data']['image']）

# 获取所有标签名
labels_attrs = self.label_interface.get_control(from_name).labels_attrs
label_names = list(labels_attrs.keys())  # ['Cat', 'Dog', ...]
```

**常见控件类型名称：**

| Label Studio 控件 | 类型字符串 | 数据对象 |
|---|---|---|
| `<RectangleLabels>` | `'RectangleLabels'` | `'Image'` |
| `<BrushLabels>` | `'BrushLabels'` | `'Image'` |
| `<KeyPointLabels>` | `'KeyPointLabels'` | `'Image'` |
| `<PolygonLabels>` | `'PolygonLabels'` | `'Image'` |
| `<Choices>` | `'Choices'` | `'Text'` / `'Image'` |

---

## 八、关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LABEL_STUDIO_URL` | — | Label Studio 地址，Docker 内不能用 `localhost` |
| `LABEL_STUDIO_API_KEY` | — | Legacy Token，Docker 模式访问图片必须 |
| `MODEL_DIR` | `.` | 模型文件目录，同时是 SQLite 缓存存放位置 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `WORKERS` | `1` | gunicorn worker 数量 |
| `THREADS` | `8` | gunicorn 线程数量 |

> **注意**：`LABEL_STUDIO_API_KEY` 只支持 Legacy Token，不支持 Personal Token（会返回 401）。

---

## 九、文件结构

```
my_backend/
├── model.py               # 核心：继承 LabelStudioMLBase，实现 predict
├── _wsgi.py               # 启动入口
├── requirements.txt       # 模型相关依赖（torch、ultralytics 等）
├── requirements-base.txt  # 框架依赖（gunicorn、label-studio-ml）
├── Dockerfile
└── docker-compose.yml
```

**`requirements-base.txt`（固定内容）：**
```
gunicorn==22.0.0
label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git
```

**`requirements.txt`（你的模型依赖）：**
```
label_studio_converter
torch>=2.0
# ultralytics
# segment-anything
# transformers
```

---

## 十、Docker 配置

**`Dockerfile`：**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["./start.sh"]
```

**`start.sh`：**
```bash
#!/bin/sh
exec gunicorn \
    --bind 0.0.0.0:${PORT:-9090} \
    --workers ${WORKERS:-1} \
    --threads ${THREADS:-8} \
    --timeout 120 \
    '_wsgi:app'
```

**`docker-compose.yml`：**
```yaml
version: "3.8"
services:
  my_backend:
    build: .
    ports:
      - "9090:9090"
    environment:
      - LOG_LEVEL=DEBUG
      - WORKERS=1
      - THREADS=8
      - MODEL_DIR=/data/models
      - LABEL_STUDIO_URL=http://your-host-ip:8080   # 不能用 localhost
      - LABEL_STUDIO_API_KEY=                        # Label Studio Legacy Token
      - MODEL_PATH=/data/models/model.pth
    volumes:
      - "./data/server:/data"
      - "./models:/data/models"
```

---

## 十一、最小可运行示例

不依赖任何模型，直接验证整个链路是否通畅：

```python
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

class EchoBackend(LabelStudioMLBase):
    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, value = self.get_first_tag_occurence('RectangleLabels', 'Image')
        return ModelResponse(predictions=[
            {
                'result': [{
                    'id': 'test',
                    'type': 'rectanglelabels',
                    'from_name': from_name,
                    'to_name': to_name,
                    'original_width': 100,
                    'original_height': 100,
                    'image_rotation': 0,
                    'value': {
                        'x': 10, 'y': 10,
                        'width': 20, 'height': 20,
                        'rotation': 0,
                        'rectanglelabels': ['Test'],
                    },
                    'score': 1.0,
                }],
                'score': 1.0,
            }
            for _ in tasks
        ])
```

启动并验证：
```bash
python _wsgi.py
curl http://localhost:9090/health
# {"status": "UP", "model_class": "EchoBackend"}
```
