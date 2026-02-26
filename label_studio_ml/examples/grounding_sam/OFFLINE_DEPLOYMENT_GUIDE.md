# Grounding SAM 离线部署完整指南

**作者贡献**：本指南补充了 Grounding SAM 后端的离线部署方案

适用于 Docker 镜像：`heartexlabs/label-studio-ml-backend:grounding_sam-master`

## 目录

- [快速开始](#快速开始)
- [关键发现：BERT 模型依赖](#关键发现bert-模型依赖)
- [完整依赖清单](#完整依赖清单)
- [模型准备](#模型准备)
  - [GroundingDINO 模型](#groundingdino-模型)
  - [BERT 文本编码器](#bert-文本编码器)
  - [SAM 分割模型](#sam-分割模型)
- [BERT 下载替代方案](#bert-下载替代方案)
- [Docker 容器配置](#docker-容器配置)
- [离线部署步骤](#离线部署步骤)
- [验证和测试](#验证和测试)
- [故障排查](#故障排查)
- [最佳实践](#最佳实践)

---

## 快速开始

如果你急于开始，这里是最简化的步骤：

```bash
# 1. 准备模型文件
mkdir -p ~/grounding_sam_models/huggingface_cache
cd ~/grounding_sam_models

# 2. 下载 GroundingDINO 权重 (600MB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 3. 下载 SAM 权重 (2.4GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 4. 下载 BERT 模型缓存 (440MB) - 使用镜像站
export HF_ENDPOINT='https://hf-mirror.com'
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModel
import os
cache_dir = os.path.expanduser("~/grounding_sam_models/huggingface_cache")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
model = AutoModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
print("✅ BERT 下载完成")
EOF

# 5. 启动容器
docker run -d \
  --name grounding_sam \
  -p 9090:9090 \
  -v ~/grounding_sam_models/groundingdino_swint_ogc.pth:/GroundingDINO/weights/groundingdino_swint_ogc.pth:ro \
  -v ~/grounding_sam_models/sam_vit_h_4b8939.pth:/app/sam_vit_h_4b8939.pth:ro \
  -v ~/grounding_sam_models/huggingface_cache:/root/.cache/huggingface:ro \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e USE_SAM=true \
  -e USE_MOBILE_SAM=false \
  heartexlabs/label-studio-ml-backend:grounding_sam-master

# 6. 验证
curl http://localhost:9090/health
```

---

## 关键发现：BERT 模型依赖

⚠️ **重要**：GroundingDINO 使用 **BERT-base-uncased** 作为文本编码器，这个模型在**首次运行时**会自动从 HuggingFace 下载。

**如果容器内无网络访问，会导致启动失败！**

### GroundingDINO 实际依赖 3 类模型

1. ✅ **GroundingDINO 视觉模型** (600MB) - 物体检测
2. ⚠️ **BERT 文本编码器** (440MB) - 将 "cat", "dog" 等文本转为向量
3. ✅ **SAM 分割模型** (2.4GB 或 40MB) - 精确分割（可选）

### BERT 模型信息

| 项目 | 详情 |
|------|------|
| **模型名称** | bert-base-uncased |
| **来源** | HuggingFace Hub |
| **大小** | 约 440MB |
| **下载时机** | 首次运行 GroundingDINO 时 |
| **缓存位置** | `/root/.cache/huggingface/hub/` |
| **用途** | 文本提示编码 |

---

## 完整依赖清单

### 必需文件

#### 1. GroundingDINO 权重（必需）
- **文件**：`groundingdino_swint_ogc.pth`
- **大小**：600MB
- **下载**：https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
- **容器路径**：`/GroundingDINO/weights/groundingdino_swint_ogc.pth`

#### 2. BERT 模型缓存（必需）⚠️
- **目录**：`huggingface_cache/`
- **大小**：约 440MB
- **模型**：bert-base-uncased
- **容器路径**：`/root/.cache/huggingface/`

#### 3. SAM 权重（可选，根据配置）
- **标准 SAM**：`sam_vit_h_4b8939.pth` (2.4GB)
  - 下载：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  - 容器路径：`/app/sam_vit_h_4b8939.pth`
- **或 MobileSAM**：`mobile_sam.pt` (40MB)
  - 下载：https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
  - 容器路径：`/app/mobile_sam.pt`

### 总大小

- **最小配置**：约 1GB (GroundingDINO + BERT)
- **推荐配置**：约 3.4GB (GroundingDINO + BERT + 标准 SAM)
- **轻量配置**：约 1.1GB (GroundingDINO + BERT + MobileSAM)

### 容器内已包含（无需准备）

预构建镜像已包含：
- ✅ GroundingDINO 代码仓库（`/GroundingDINO/groundingdino/`）
- ✅ 配置文件（`/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`）
- ✅ Python 依赖（transformers, torch, 等）

⚠️ **注意**：不要用 volume 覆盖 `/GroundingDINO/` 根目录，否则会丢失代码和配置文件！

---

## 模型准备

### GroundingDINO 模型

```bash
mkdir -p ~/grounding_sam_models
cd ~/grounding_sam_models

# 使用 wget
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 或使用 curl
curl -L -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### BERT 文本编码器

BERT 模型需要特殊处理，因为它以 HuggingFace 缓存格式存储。

#### 方法 1：使用 HuggingFace 镜像站（推荐）⭐

```bash
mkdir -p ~/grounding_sam_models/huggingface_cache

# 使用国内镜像站
export HF_ENDPOINT='https://hf-mirror.com'

python3 << 'EOF'
from transformers import AutoTokenizer, AutoModel
import os

cache_dir = os.path.expanduser("~/grounding_sam_models/huggingface_cache")
print(f"下载到: {cache_dir}")

print("下载 BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    cache_dir=cache_dir
)

print("下载 BERT 模型...")
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    cache_dir=cache_dir
)

print("✅ BERT 模型下载完成！")
EOF
```

#### 方法 2：使用代理

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897

python3 << 'EOF'
from transformers import AutoTokenizer, AutoModel
import os

cache_dir = os.path.expanduser("~/grounding_sam_models/huggingface_cache")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    cache_dir=cache_dir,
    resume_download=True
)

model = AutoModel.from_pretrained(
    "bert-base-uncased",
    cache_dir=cache_dir,
    resume_download=True
)

print("✅ BERT 模型下载完成！")
EOF
```

#### 方法 3：使用 ModelScope（阿里达摩院）

```bash
pip install modelscope

python3 << 'EOF'
from modelscope.hub.snapshot_download import snapshot_download
cache_dir = '~/grounding_sam_models/huggingface_cache'
model_dir = snapshot_download('damo/nlp_bert_base-uncased', cache_dir=cache_dir)
print(f"✅ 模型下载到: {model_dir}")
EOF
```

### SAM 分割模型

#### 标准 SAM (推荐高精度场景)

```bash
cd ~/grounding_sam_models

# 使用 wget
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 或使用 curl
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### MobileSAM (推荐资源受限场景)

```bash
cd ~/grounding_sam_models

# 使用 wget
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# 或使用 curl
curl -L -o mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

### 验证下载

```bash
ls -lh ~/grounding_sam_models

# 预期输出：
# -rw-r--r-- groundingdino_swint_ogc.pth  (600MB)
# -rw-r--r-- sam_vit_h_4b8939.pth         (2.4GB)
# drwxr-xr-x huggingface_cache/           (440MB)

# 验证 BERT 缓存结构
ls -R ~/grounding_sam_models/huggingface_cache/

# 应该看到：
# models--bert-base-uncased/
#   ├── blobs/
#   ├── refs/
#   └── snapshots/
```

---

## BERT 下载替代方案

如果上述 BERT 下载方法都失败，以下是替代方案：

### 方案 1：从其他机器复制

如果你有另一台可以联网的机器：

```bash
# 在有网络的机器上
mkdir -p ~/bert_cache
export HF_HOME=~/bert_cache
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"

# 打包
cd ~
tar -czf bert_cache.tar.gz bert_cache/

# 传输到目标机器后解压
tar -xzf bert_cache.tar.gz
mkdir -p ~/grounding_sam_models/huggingface_cache
mv bert_cache/* ~/grounding_sam_models/huggingface_cache/
```

### 方案 2：手动下载文件

访问 HuggingFace 镜像站手动下载：

1. 浏览器访问：https://hf-mirror.com/bert-base-uncased/tree/main
2. 下载以下文件：
   - `config.json` (0.5KB)
   - `vocab.txt` (226KB)
   - `tokenizer.json` (466KB)
   - `tokenizer_config.json` (0.5KB)
   - `pytorch_model.bin` (440MB)

3. 组织目录结构：

```bash
MODEL_DIR=~/grounding_sam_models/huggingface_cache
BERT_DIR=$MODEL_DIR/models--bert-base-uncased
SNAPSHOT_DIR=$BERT_DIR/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594

mkdir -p $SNAPSHOT_DIR
mkdir -p $BERT_DIR/refs

# 将下载的文件放到 snapshot 目录
mv config.json vocab.txt tokenizer.json tokenizer_config.json pytorch_model.bin $SNAPSHOT_DIR/

# 创建引用
echo "86b5e0934494bd15c9632b12f734a8a67f723594" > $BERT_DIR/refs/main
```

### 方案 3：使用预打包资源

搜索关键词：`bert-base-uncased huggingface cache`，从网盘等渠道获取预打包的缓存文件。

---

## Docker 容器配置

### 环境变量说明

| 变量名 | 说明 | 默认值 | 推荐值 |
|--------|------|--------|--------|
| `USE_SAM` | 是否启用 SAM 分割 | `false` | `true` |
| `USE_MOBILE_SAM` | 是否使用 MobileSAM | `false` | `false` (标准SAM) |
| `BOX_THRESHOLD` | 边界框置信度阈值 | `0.30` | `0.30` |
| `TEXT_THRESHOLD` | 文本匹配阈值 | `0.25` | `0.25` |
| `TRANSFORMERS_OFFLINE` | Transformers 离线模式 | - | `1` |
| `HF_DATASETS_OFFLINE` | HuggingFace 离线模式 | - | `1` |
| `WORKERS` | 工作进程数 | `2` | `2-4` |
| `THREADS` | 线程数 | `4` | `4` |

### Volume 映射说明

#### ✅ 正确方式：只映射权重文件（推荐）

```bash
-v ~/grounding_sam_models/groundingdino_swint_ogc.pth:/GroundingDINO/weights/groundingdino_swint_ogc.pth:ro
-v ~/grounding_sam_models/sam_vit_h_4b8939.pth:/app/sam_vit_h_4b8939.pth:ro
-v ~/grounding_sam_models/huggingface_cache:/root/.cache/huggingface:ro
```

**优势**：
- ✅ GroundingDINO 代码和配置仍在容器内
- ✅ 只替换权重文件
- ✅ 简单且稳定

#### ❌ 错误方式：覆盖整个目录

```bash
-v ~/my_models:/GroundingDINO  # ❌ 会丢失代码和配置文件！
```

**后果**：配置文件和代码丢失，容器无法启动

---

## 离线部署步骤

### 场景：机器 A（有网络）→ 机器 B（无网络）

#### 步骤 1：在机器 A 准备所有文件

```bash
mkdir -p ~/grounding_sam_offline
cd ~/grounding_sam_offline

# 下载 GroundingDINO 权重
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 下载 SAM 权重
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 下载 BERT 缓存
mkdir -p huggingface_cache
export HF_ENDPOINT='https://hf-mirror.com'
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModel
import os
cache_dir = os.path.expanduser("~/grounding_sam_offline/huggingface_cache")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
model = AutoModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
print("✅ BERT 下载完成")
EOF

# 验证所有文件
ls -lh
ls -R huggingface_cache/
```

#### 步骤 2：打包传输

```bash
cd ~/grounding_sam_offline
tar -czf grounding_sam_models.tar.gz \
    groundingdino_swint_ogc.pth \
    sam_vit_h_4b8939.pth \
    huggingface_cache/

# 检查打包文件
ls -lh grounding_sam_models.tar.gz
# 预期大小：约 2.5-3GB（压缩后）

# 通过 U盘、scp、或共享文件夹传输到机器 B
```

#### 步骤 3：在机器 B 解压

```bash
# 在机器 B 上
cd ~
tar -xzf grounding_sam_models.tar.gz

# 验证解压
ls -lh groundingdino_swint_ogc.pth
ls -lh sam_vit_h_4b8939.pth
ls -R huggingface_cache/
```

#### 步骤 4：在机器 B 启动容器（离线）

```bash
docker run -d \
  --name grounding_sam \
  --restart unless-stopped \
  -p 9090:9090 \
  \
  -v ~/groundingdino_swint_ogc.pth:/GroundingDINO/weights/groundingdino_swint_ogc.pth:ro \
  -v ~/sam_vit_h_4b8939.pth:/app/sam_vit_h_4b8939.pth:ro \
  -v ~/huggingface_cache:/root/.cache/huggingface:ro \
  \
  -e USE_SAM=true \
  -e USE_MOBILE_SAM=false \
  -e BOX_THRESHOLD=0.30 \
  -e TEXT_THRESHOLD=0.25 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  \
  heartexlabs/label-studio-ml-backend:grounding_sam-master
```

---

## 验证和测试

### 1. 检查容器状态

```bash
# 查看容器运行状态
docker ps | grep grounding_sam

# 查看启动日志
docker logs grounding_sam

# 成功的日志应包含：
# INFO - Loading SAM model...
# INFO - SAM model successfully loaded!
# INFO - Starting server on port 9090
```

### 2. 验证模型文件

```bash
# 检查 GroundingDINO 权重
docker exec grounding_sam ls -lh /GroundingDINO/weights/
# 应该看到：groundingdino_swint_ogc.pth (600MB)

# 检查 SAM 权重
docker exec grounding_sam ls -lh /app/sam_vit_h_4b8939.pth
# 应该看到：sam_vit_h_4b8939.pth (2.4GB)

# 检查 BERT 缓存
docker exec grounding_sam ls -lh /root/.cache/huggingface/
# 应该看到：models--bert-base-uncased/

# 检查配置文件（确保没有被覆盖）
docker exec grounding_sam cat /GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py | head -20
```

### 3. 验证 BERT 离线加载

```bash
# 检查日志中是否有 BERT 相关信息
docker logs grounding_sam 2>&1 | grep -i "bert\|transformer\|token"

# 成功应该看到类似：
# Loading BERT tokenizer from cache
# Using bert-base-uncased
# 不应该看到 "Downloading" 相关消息

# 测试 BERT 加载
docker exec grounding_sam python3 << 'EOF'
from transformers import AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'

try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"✅ BERT 离线加载成功 (vocab_size: {tokenizer.vocab_size})")
except Exception as e:
    print(f"❌ BERT 加载失败: {e}")
EOF
```

### 4. API 健康检查

```bash
# 健康检查
curl http://localhost:9090/health
# 预期输出：{"status":"UP"}

# 测试推理（可选，需要准备图片 URL）
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "data": {
        "image": "https://example.com/test.jpg"
      }
    }],
    "params": {
      "prompt": "cat"
    }
  }'
```

### 5. 离线测试（确认无网络依赖）

```bash
# 停止容器
docker stop grounding_sam
docker rm grounding_sam

# 以无网络模式启动
docker run -d \
  --name grounding_sam \
  --network none \
  -p 9090:9090 \
  -v ~/groundingdino_swint_ogc.pth:/GroundingDINO/weights/groundingdino_swint_ogc.pth:ro \
  -v ~/sam_vit_h_4b8939.pth:/app/sam_vit_h_4b8939.pth:ro \
  -v ~/huggingface_cache:/root/.cache/huggingface:ro \
  -e USE_SAM=true \
  -e USE_MOBILE_SAM=false \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  heartexlabs/label-studio-ml-backend:grounding_sam-master

# 查看日志，不应该有任何网络错误
docker logs grounding_sam 2>&1 | grep -i "download\|network\|connection"
```

---

## 故障排查

### 问题 1：容器启动失败 - "No such file: GroundingDINO_SwinT_OGC.py"

**原因**：GroundingDINO 配置文件丢失

**检查**：
```bash
docker exec grounding_sam ls -lh /GroundingDINO/groundingdino/config/
```

**解决**：
- 不要用 volume 覆盖 `/GroundingDINO/` 根目录
- 只映射权重文件：`/GroundingDINO/weights/groundingdino_swint_ogc.pth`

### 问题 2：容器启动卡住 - 日志显示 "Downloading bert-base-uncased"

**原因**：BERT 模型未预下载，容器尝试在线下载但无网络

**解决**：
1. 停止容器：`docker stop grounding_sam`
2. 准备 BERT 缓存（参考 [BERT 文本编码器](#bert-文本编码器)）
3. 重新启动，添加 BERT 缓存映射和离线环境变量

### 问题 3：错误 "OSError: Can't load tokenizer for 'bert-base-uncased'"

**原因**：BERT 缓存路径映射不正确或缓存目录为空

**检查**：
```bash
# 检查本地缓存
ls -R ~/grounding_sam_models/huggingface_cache/

# 应该有结构：
# models--bert-base-uncased/
#   ├── blobs/
#   ├── refs/
#   └── snapshots/

# 检查容器内映射
docker exec grounding_sam ls -R /root/.cache/huggingface/
```

**解决**：
- 确保本地 BERT 缓存目录结构正确
- 检查 volume 映射路径拼写
- 验证文件权限（应该可读）

### 问题 4：错误 "checkpoint file not found"

**原因**：权重文件路径不正确

**检查**：
```bash
# 查看容器内路径
docker exec grounding_sam ls -lh /GroundingDINO/weights/
docker exec grounding_sam ls -lh /app/

# 查看环境变量
docker exec grounding_sam env | grep -E "CHECKPOINT|GROUNDINGDINO"
```

**解决**：
- 确保 volume 映射路径正确
- 文件名拼写正确（区分大小写）
- 文件权限为可读（chmod 644）

### 问题 5：推理速度很慢

**优化建议**：
1. 使用 GPU 加速：添加 `--gpus all` 参数
2. 使用 MobileSAM 替代标准 SAM：`-e USE_MOBILE_SAM=true`
3. 调低阈值减少检测数量：`-e BOX_THRESHOLD=0.35`
4. 增加工作进程：`-e WORKERS=4`

### 问题 6：如何确认 BERT 是从本地加载？

**方法 1：断网测试**
```bash
docker run -d \
  --name grounding_sam \
  --network none \
  -v ... \
  ...
```

**方法 2：查看日志**
```bash
docker logs grounding_sam 2>&1 | grep -i download
# 不应该看到任何 "Downloading" 消息
```

---

## 最佳实践

### 推荐配置（生产环境）

#### CPU 版本

```bash
docker run -d \
  --name grounding_sam \
  --restart unless-stopped \
  -p 9090:9090 \
  \
  -v ~/grounding_sam_models/groundingdino_swint_ogc.pth:/GroundingDINO/weights/groundingdino_swint_ogc.pth:ro \
  -v ~/grounding_sam_models/sam_vit_h_4b8939.pth:/app/sam_vit_h_4b8939.pth:ro \
  -v ~/grounding_sam_models/huggingface_cache:/root/.cache/huggingface:ro \
  \
  -e USE_SAM=true \
  -e USE_MOBILE_SAM=false \
  -e BOX_THRESHOLD=0.30 \
  -e TEXT_THRESHOLD=0.25 \
  -e WORKERS=4 \
  -e THREADS=4 \
  -e LOG_LEVEL=INFO \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  \
  heartexlabs/label-studio-ml-backend:grounding_sam-master
```

#### GPU 版本

```bash
docker run -d \
  --name grounding_sam \
  --restart unless-stopped \
  --gpus all \
  -p 9090:9090 \
  \
  -v ~/grounding_sam_models/groundingdino_swint_ogc.pth:/GroundingDINO/weights/groundingdino_swint_ogc.pth:ro \
  -v ~/grounding_sam_models/sam_vit_h_4b8939.pth:/app/sam_vit_h_4b8939.pth:ro \
  -v ~/grounding_sam_models/huggingface_cache:/root/.cache/huggingface:ro \
  \
  -e USE_SAM=true \
  -e USE_MOBILE_SAM=false \
  -e BOX_THRESHOLD=0.30 \
  -e TEXT_THRESHOLD=0.25 \
  -e WORKERS=2 \
  -e THREADS=4 \
  -e LOG_LEVEL=INFO \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  \
  heartexlabs/label-studio-ml-backend:grounding_sam-master
```

### 快速启动脚本

保存为 `start_grounding_sam_offline.sh`：

```bash
#!/bin/bash
# Grounding SAM 离线启动脚本

set -e

MODEL_DIR=~/grounding_sam_models

echo "=========================================="
echo "Grounding SAM 离线启动检查"
echo "=========================================="

# 检查模型文件
echo ""
echo "[1/3] 检查 GroundingDINO 模型..."
if [ ! -f "$MODEL_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "❌ 错误：找不到 GroundingDINO 模型文件"
    echo "   位置：$MODEL_DIR/groundingdino_swint_ogc.pth"
    echo "   下载：https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    exit 1
fi
echo "✅ GroundingDINO 模型文件存在"

echo ""
echo "[2/3] 检查 SAM 模型..."
if [ ! -f "$MODEL_DIR/sam_vit_h_4b8939.pth" ]; then
    echo "❌ 错误：找不到 SAM 模型文件"
    echo "   位置：$MODEL_DIR/sam_vit_h_4b8939.pth"
    echo "   下载：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    exit 1
fi
echo "✅ SAM 模型文件存在"

echo ""
echo "[3/3] 检查 BERT 缓存..."
if [ ! -d "$MODEL_DIR/huggingface_cache/models--bert-base-uncased" ]; then
    echo "❌ 错误：找不到 BERT 模型缓存"
    echo "   位置：$MODEL_DIR/huggingface_cache/"
    echo "   请参考文档下载 BERT 模型"
    exit 1
fi
echo "✅ BERT 缓存目录存在"

echo ""
echo "=========================================="
echo "所有模型检查通过！"
echo "=========================================="

# 停止旧容器
echo ""
echo "停止旧容器（如果存在）..."
docker stop grounding_sam 2>/dev/null || true
docker rm grounding_sam 2>/dev/null || true

# 启动新容器
echo ""
echo "启动 Grounding SAM 容器..."
docker run -d \
  --name grounding_sam \
  --restart unless-stopped \
  -p 9090:9090 \
  \
  -v $MODEL_DIR/groundingdino_swint_ogc.pth:/GroundingDINO/weights/groundingdino_swint_ogc.pth:ro \
  -v $MODEL_DIR/sam_vit_h_4b8939.pth:/app/sam_vit_h_4b8939.pth:ro \
  -v $MODEL_DIR/huggingface_cache:/root/.cache/huggingface:ro \
  \
  -e USE_SAM=true \
  -e USE_MOBILE_SAM=false \
  -e BOX_THRESHOLD=0.30 \
  -e TEXT_THRESHOLD=0.25 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  \
  heartexlabs/label-studio-ml-backend:grounding_sam-master

echo ""
echo "等待容器启动..."
sleep 5

# 验证
echo ""
echo "=========================================="
echo "验证结果"
echo "=========================================="

echo ""
echo "容器状态："
docker ps | grep grounding_sam

echo ""
echo "最新日志："
docker logs --tail 20 grounding_sam

echo ""
echo "健康检查："
curl -s http://localhost:9090/health

echo ""
echo ""
echo "=========================================="
echo "✅ Grounding SAM 启动完成！"
echo "=========================================="
echo ""
echo "API 地址：http://localhost:9090"
echo "健康检查：curl http://localhost:9090/health"
echo "查看日志：docker logs -f grounding_sam"
```

使用方式：

```bash
chmod +x start_grounding_sam_offline.sh
./start_grounding_sam_offline.sh
```

### 性能对比

| 配置 | 内存需求 | 推理速度 | 精度 | 文件大小 | 推荐场景 |
|------|---------|---------|------|----------|----------|
| 仅 GroundingDINO + BERT | 3GB | 快 | 中 | 1GB | 只需检测框 |
| + MobileSAM | 4GB | 中 | 高 | 1.1GB | 平衡性能 |
| + 标准 SAM | 8GB+ | 慢 | 很高 | 3.4GB | 高精度需求 |
| + GPU | 4-8GB VRAM | 很快 | 很高 | 3.4GB | 生产环境 |

### 离线准备清单

在离线部署前，确保准备了：

- [ ] **必需**：groundingdino_swint_ogc.pth (600MB)
- [ ] **必需**：BERT huggingface_cache (440MB)
- [ ] **推荐**：sam_vit_h_4b8939.pth (2.4GB) - 标准 SAM
- [ ] **可选**：mobile_sam.pt (40MB) - MobileSAM
- [ ] **确认**：Docker 镜像已拉取到本地
- [ ] **确认**：volume 映射只指向权重文件，不覆盖代码目录
- [ ] **确认**：设置了离线环境变量（TRANSFORMERS_OFFLINE=1）

---

## 总结

### 关键点

1. **GroundingDINO 依赖 3 类模型**：
   - GroundingDINO 视觉模型（检测）
   - BERT 文本模型（文本编码）← **容易遗漏！**
   - SAM 分割模型（可选）

2. **BERT 必须提前下载**：
   - 首次运行会从 HuggingFace 自动下载
   - 离线环境必须提前准备缓存
   - 缓存目录映射到 `/root/.cache/huggingface/`

3. **Volume 映射最佳实践**：
   - ✅ **推荐**：只映射权重文件
   - ⚠️ **谨慎**：映射整个 GroundingDINO 目录（需完整准备）
   - ❌ **禁止**：映射空目录覆盖 `/GroundingDINO/`

4. **离线模式环境变量**：
   - `TRANSFORMERS_OFFLINE=1` - 禁止在线下载
   - `HF_DATASETS_OFFLINE=1` - 禁用在线检查

### 文件大小总结

| 文件/目录 | 大小 | 必需 | 说明 |
|----------|------|------|------|
| groundingdino_swint_ogc.pth | 600MB | ✅ 是 | GroundingDINO 权重 |
| huggingface_cache (BERT) | 440MB | ✅ 是 | BERT 文本编码器 |
| sam_vit_h_4b8939.pth | 2.4GB | ⚠️ 可选 | 标准 SAM（高精度） |
| mobile_sam.pt | 40MB | ⚠️ 可选 | MobileSAM（轻量级） |
| **最小配置总计** | **1GB** | | GroundingDINO + BERT |
| **推荐配置总计** | **3.4GB** | | + 标准 SAM |

---

## 参考资源

- [GroundingDINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
- [MobileSAM GitHub](https://github.com/ChaoningZhang/MobileSAM)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Label Studio ML Backend](https://github.com/HumanSignal/label-studio-ml-backend)

---

## 贡献

本文档由社区贡献者补充，旨在帮助在离线或受限网络环境中部署 Grounding SAM。

**最后更新**：2024-02-26

**贡献者**：@kitemanuel

如有问题或改进建议，欢迎提交 Issue 或 Pull Request。
