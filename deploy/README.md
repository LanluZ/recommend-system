## 轻量化推理部署指南

本目录提供基于 **ONNX Runtime** 的轻量化推理解决方案，支持 CPU/GPU 高效推理，无需 PyTorch。

### 文件说明

| 文件 | 功能 | 说明 |
|------|------|------|
| `inference.py` | 核心推理引擎 | 提供 `RecommendationEngine` 类，支持单用户和批量推荐 |
| `cli.py` | 命令行工具 | 快速查询推荐，支持单用户、批量、列表操作 |
| `api.py` | Flask HTTP API | 微服务接口，支持 REST 调用 |
| `requirements-deploy.txt` | 依赖清单 | 部署所需的最小依赖 |

---

### 快速开始

#### 1. 安装依赖

```bash
# 仅需 ONNX Runtime（不需要 PyTorch）
pip install -r requirements-deploy.txt
```

**最小依赖：**
- `onnxruntime>=1.15.0` - 轻量化推理框架
- `flask>=2.0.0` - API 服务（可选）

#### 2. 使用核心推理模块

```python
from inference import RecommendationEngine

# 初始化引擎
engine = RecommendationEngine(
    model_path="outputs/model.onnx",
    dataset_path="data/processed/dataset.json",
    use_gpu=False,  # 使用 GPU：True
)

# 单用户推荐
history = [1, 2, 3, 5, 7]
recommendations = engine.recommend(history, topk=10, return_scores=True)
for rec in recommendations:
    print(f"{rec['title']} - {rec['score']:.4f}")

# 批量推荐
histories = [[1, 2, 3], [4, 5, 6]]
batch_results = engine.batch_recommend(histories, topk=5)
```

#### 3. 命令行工具

```bash
# 查看所有用户
python -m deploy.cli --list-users

# 为用户生成推荐
python -m deploy.cli --user "Kaito" --topk 10 --output result.json

# 批量推荐（从文件读取用户列表）
python -m deploy.cli --batch users.json --topk 5

# 使用 GPU 推理
python -m deploy.cli --user "Kaito" --gpu
```

**命令行参数：**
- `--user <用户名>` - 单用户推荐
- `--batch <文件>` - 批量推荐（JSON 格式）
- `--topk <数字>` - 推荐数量（默认 10）
- `--list-users` - 列出所有用户
- `--output <文件>` - 输出结果到 JSON 文件
- `--gpu` - 启用 GPU 推理
- `--model <路径>` - 指定 ONNX 模型路径
- `--dataset <路径>` - 指定数据集路径

#### 4. Flask API 服务

```bash
# 启动 API 服务（默认：http://127.0.0.1:5000）
python -m deploy.api

# 使用 GPU
python -m deploy.api --gpu

# 自定义端口
python -m deploy.api --host 0.0.0.0 --port 8000
```

---

### API 调用示例

#### 健康检查

```bash
curl http://127.0.0.1:5000/health
```

**返回：**
```json
{
  "status": "healthy",
  "model": "SASRec",
  "num_items": 1000
}
```

#### 列表用户

```bash
curl http://127.0.0.1:5000/users
```

**返回：**
```json
{
  "count": 5,
  "users": ["Kaito", "Alice", "Bob", ...]
}
```

#### 单用户推荐

```bash
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user": "Kaito",
    "topk": 10
  }'
```

**或使用历史序列：**

```bash
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "history": [1, 2, 3, 5, 7],
    "topk": 5
  }'
```

**返回：**
```json
{
  "user": "Kaito",
  "history": [1, 2, 3, 5, 7],
  "topk": 10,
  "recommendations": [
    {
      "id": 10,
      "title": "电影标题1",
      "score": 3.245
    },
    {
      "id": 15,
      "title": "电影标题2",
      "score": 3.102
    }
  ]
}
```

#### 批量推荐

```bash
curl -X POST http://127.0.0.1:5000/recommend/batch \
  -H "Content-Type: application/json" \
  -d '{
    "histories": {
      "Kaito": [1, 2, 3],
      "Alice": [4, 5, 6]
    },
    "topk": 5
  }'
```

**返回：**
```json
{
  "count": 2,
  "topk": 5,
  "results": {
    "Kaito": [...],
    "Alice": [...]
  }
}
```

---

### 使用场景

#### 场景 1：离线批量推荐

适合定期生成大量推荐的场景。

```bash
# 生成所有用户的推荐
python -m deploy.cli --list-users > users.txt
# 手动编辑 users.json，包含所有用户名
python -m deploy.cli --batch users.json --topk 10 --output recommendations.json
```

#### 场景 2：在线实时服务

使用 API 服务，支持实时推荐查询。

```bash
# 启动 API
python -m deploy.api --host 0.0.0.0 --port 5000

# 客户端调用
curl -X POST http://server:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user": "user123", "topk": 10}'
```

#### 场景 3：嵌入应用代码

直接导入 `RecommendationEngine` 到你的应用。

```python
from deploy.inference import RecommendationEngine

engine = RecommendationEngine(...)
recs = engine.recommend([1, 2, 3], topk=10)
```

---

### 性能特点

| 特性 | 说明 |
|------|------|
| **轻量化** | 仅需 ONNX Runtime，无 PyTorch 依赖，包体积小 |
| **快速推理** | CPU 推理典型延迟 < 10ms；GPU 推理 < 5ms |
| **批量处理** | 支持批量推理，加速多用户查询 |
| **跨平台** | CPU/GPU 推理均支持，兼容 Linux/Windows/MacOS |
| **易于部署** | 可部署为 Docker 容器或 K8s 服务 |

---

### 配置说明

#### 使用 GPU 推理

```python
# 需要 ONNX Runtime GPU 版本
# pip install onnxruntime-gpu

engine = RecommendationEngine(
    model_path="outputs/model.onnx",
    dataset_path="data/processed/dataset.json",
    use_gpu=True,  # 启用 GPU
)
```

#### 自定义模型路径

```bash
python -m deploy.cli \
  --model /path/to/model.onnx \
  --dataset /path/to/dataset.json \
  --user "Kaito"
```

---

### 进阶用法

#### 自定义推理参数

```python
engine = RecommendationEngine(...)

# 不过滤历史物品
recommendations = engine.recommend(
    history=[1, 2, 3],
    topk=10,
    filter_history=False,  # 包括历史物品
    return_scores=True,
)
```

#### 批量推理优化

```python
# 大批量推理（推荐 > 100 用户时）
histories = [[1,2,3], [4,5,6], ...]  # 1000+ 个用户
batch_size = 100

for i in range(0, len(histories), batch_size):
    batch = histories[i : i + batch_size]
    results = engine.batch_recommend(batch, topk=10)
    # 处理 results
```

#### Docker 部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -r deploy/requirements-deploy.txt

CMD ["python", "-m", "deploy.api", "--host", "0.0.0.0", "--port", "5000"]
```

构建并运行：

```bash
docker build -t recommend-api .
docker run -p 5000:5000 recommend-api
```

---

### 总结

| 方式 | 优点 | 场景 |
|------|------|------|
| **Python API** | 灵活、易于集成 | 应用程序内部调用 |
| **命令行工具** | 简单易用 | 脚本、离线处理 |
| **HTTP API** | 通用、跨语言 | 微服务、分布式系统 |
