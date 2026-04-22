# recommend-system

基于 PyTorch + Transformer（SASRec）的电影序列推荐示例，支持：

- 检查并清洗爬取数据
- 调整目录结构（`data/raw`、`data/processed`、`src`、`outputs`）
- 输入不定长历史（最大 50 条）
- 输出推荐电影列表

## 目录结构

```text
data/
  raw/clean.csv
  processed/dataset.json
src/
  preprocess.py
  model.py
  train.py
  recommend.py
  visualize.py
outputs/
  model.pt
  model.onnx
  training_params.json
  loss_curve.png
  model_summary.txt
```

## 快速开始

### 1) 安装依赖

环境要求：Python >= 3.10，推荐使用 conda 虚拟环境。

```bash
# 仅声明依赖版本（若已在 conda 环境中可跳过）
pip install -r requirements.txt
```

主要依赖：
- `torch>=2.2.0` - PyTorch（含 CUDA 支持可选）
- `numpy>=1.26.0` - 数值计算
- `onnx>=1.16.0` - 模型格式支持
- `matplotlib>=3.5.0` - 图表绘制
- `netron` - 模型结构可视化

### 2) 数据预处理

检查并清洗原始数据，生成训练集：

```bash
python -m src.preprocess --input data/raw/clean.csv --output-dir data/processed
```

输入要求：CSV 文件，包含字段 `电影id`、`标题`、`用户名`。

产物：
- `data/processed/dataset.json`：包含训练/验证/测试分割的数据集

### 3) 模型训练

使用 SASRec 训练序列推荐模型：

```bash
python -m src.train --dataset data/processed/dataset.json --output-dir outputs --max-len 50 --weight-decay 1e-4
```

训练完成后会输出：

- `outputs/model.pt`：模型 checkpoint（含权重和超参数配置）
- `outputs/model.onnx`：ONNX 推理模型（可跨平台部署）
- `outputs/training_params.json`：完整训练日志（参数、数据统计、每轮指标、测试结果）

常用参数：
- `--epochs 20`：训练轮数
- `--batch-size 64`：批大小
- `--hidden-size 128`：隐藏维度
- `--num-layers 2`：Transformer 层数
- `--num-heads 4`：多头注意力头数

### 4) 生成推荐

基于训练好的模型为指定用户生成推荐列表：

```bash
python -m src.recommend --dataset data/processed/dataset.json --checkpoint outputs/model.pt --user "Kaito" --topk 10
```

输出：JSON 格式的推荐结果（包含电影 ID、标题）。

### 5) 结果可视化

#### 训练 Loss 曲线

生成训练过程的损失曲线和模型摘要：

```bash
python -m src.visualize --training-params outputs/training_params.json --output-dir outputs
```

产物：
- `outputs/loss_curve.png`：使用 matplotlib 绘制的 Loss 曲线（含统计信息）
- `outputs/model_summary.txt`：模型结构文字摘要

#### 模型结构（Netron 交互式查看）

使用 Netron 可视化 TorchScript 模型的计算图：

```bash
python -m src.visualize --checkpoint outputs/model.pt --netron --browse
```

效果：
- 自动从 `model.pt` 导出 TorchScript 格式（`outputs/model_netron.pt`）
- 启动本地 Netron 服务（`http://127.0.0.1:8081`）
- 打开浏览器显示交互式模型结构图
- 按 `Ctrl+C` 停止服务

## 代码组织

### 核心模块（`src/`）

所有模块都已补充中文注释，便于理解核心逻辑：

| 文件 | 功能 | 关键说明 |
|------|------|--------|
| `model.py` | SASRec 模型实现 | Transformer Encoder、因果 Mask、位置编码、序列聚合 |
| `preprocess.py` | 数据预处理 | CSV 验证、去重、编码映射、train/val/test 分割 |
| `train.py` | 训练流程 | 批数据处理、优化器、EarlyStopping、ONNX/参数导出 |
| `recommend.py` | 推荐生成 | 模型推理、历史过滤、Top-K 排序 |
| `visualize.py` | 结果可视化 | Matplotlib 绘图、TorchScript 导出、Netron 启动 |

### 训练产物（`outputs/`）

| 文件 | 说明 |
|------|------|
| `model.pt` | PyTorch checkpoint，包含权重和超参数配置 |
| `model.onnx` | ONNX 格式模型（跨框架部署用） |
| `model_netron.pt` | TorchScript 格式（Netron 可视化用） |
| `training_params.json` | 完整训练日志（参数、数据统计、每轮指标、测试结果） |
| `loss_curve.png` | Matplotlib 绘制的训练 Loss 曲线 |
| `model_summary.txt` | 模型结构文字摘要 |

## 完整工作流示例

### 场景：训练新模型并生成推荐

```bash
# 1. 数据预处理
python -m src.preprocess --input data/raw/clean.csv --output-dir data/processed

# 2. 训练模型（GPU 可用会自动使用）
python -m src.train \
  --dataset data/processed/dataset.json \
  --output-dir outputs \
  --epochs 30 \
  --batch-size 32 \
  --hidden-size 256 \
  --num-layers 4

# 3. 生成推荐（为多个用户）
python -m src.recommend --dataset data/processed/dataset.json --checkpoint outputs/model.pt --user "Kaito" --topk 10
python -m src.recommend --dataset data/processed/dataset.json --checkpoint outputs/model.pt --user "Alice" --topk 5

# 4. 查看训练过程（生成 Loss 曲线）
python -m src.visualize --training-params outputs/training_params.json --output-dir outputs

# 5. 交互式查看模型结构（浏览器打开）
python -m src.visualize --checkpoint outputs/model.pt --netron --browse
```


## 方法说明

使用行业常见的序列推荐方法 SASRec：
- `Transformer Encoder` 建模用户最近行为序列；
- 因果 Mask 保证只利用历史行为预测下一个电影；
- 采用 Top-K 指标（Hit@10 / NDCG@10）进行验证。

### 模型架构（SASRec）

- 输入：长度为 `max_len` 的电影 ID 序列（左侧 `0` padding）
- Embedding：
  - `Item Embedding`：`(num_items + 1, hidden_size)`，`0` 为 padding
  - `Position Embedding`：`(max_len, hidden_size)`
- 主干网络：
  - `num_layers` 层 `TransformerEncoderLayer`
  - 多头注意力头数 `num_heads`
  - 前馈维度 `4 * hidden_size`
  - Dropout：`dropout`
  - 使用因果 Mask + Padding Mask
- 序列聚合：取每个用户序列最后一个有效位置的隐藏状态
- 输出层：`Linear(hidden_size, num_items + 1)`，训练时使用 `CrossEntropyLoss`

### 训练相关参数

可通过命令行配置：

- `--epochs`（默认 `20`）：训练轮数
- `--batch-size`（默认 `64`）：批大小
- `--max-len`（默认 `50`）：序列最大长度
- `--hidden-size`（默认 `128`）：隐藏维度
- `--num-layers`（默认 `2`）：Transformer 层数
- `--num-heads`（默认 `4`）：多头注意力头数
- `--dropout`（默认 `0.2`）：Dropout 比例
- `--lr`（默认 `1e-3`）：学习率
- `--weight-decay`（默认 `1e-4`）：L2 正则化

`training_params.json` 中会记录：

- 训练时间、设备信息（CPU/GPU）
- 数据集统计（用户数、物品数、总样本数）
- 超参数完整配置
- 每轮训练损失与验证指标（Hit@10、NDCG@10）
- 最佳验证指标出现的轮次与最终测试指标
- 模型产物路径（`model.pt`、`model.onnx`）

## 可视化详解

### Loss 曲线生成（Matplotlib）

`visualize.py` 使用 Matplotlib 生成高质量 PNG 图表：

```bash
python -m src.visualize --training-params outputs/training_params.json --output-dir outputs
```

产生 `loss_curve.png`，特点：
- **分辨率**：150 DPI（适合文档和演示）
- **图表配置**：12×7 英寸大小，包含网格线
- **统计信息**：显示总轮数、最小损失值、最后一轮损失值
- **格式**：PNG 标准格式，兼容所有平台

### 模型结构可视化（Netron）

Netron 用于交互式查看计算图：

```bash
python -m src.visualize --checkpoint outputs/model.pt --netron --browse
```

工作流程：
1. 从 PyTorch checkpoint 导出 TorchScript 模型（`outputs/model_netron.pt`）
2. 启动本地 HTTP 服务（默认 `127.0.0.1:8081`）
3. 自动打开浏览器展示模型图
4. 可交互查看每层的输入/输出形状和参数

### 模型摘要生成

同时生成的 `model_summary.txt` 包含文字形式的模型结构信息。

## 评估指标说明

### Hit@10

在 Top-10 推荐中，目标物品是否被包含。
$$\text{Hit@10} = \frac{\text{被推荐且在Top-10中的样本数}}{\text{总测试样本数}}$$

### NDCG@10

**归一化折损累计增益**（Normalized Discounted Cumulative Gain），考虑推荐排序位置的质量指标。
$$\text{NDCG@10} = \frac{\text{DCG@10}}{\text{IDCG@10}}$$

其中 DCG 对靠前的正确推荐给予更高权重。NDCG@10 越接近 1，推荐效果越好。

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| PyTorch | ≥2.2.0 | 深度学习框架 |
| NumPy | ≥1.26.0 | 数值计算 |
| ONNX | ≥1.16.0 | 模型导出与兼容性 |
| Matplotlib | ≥3.5.0 | 数据可视化 |
| Netron | 最新 | 模型结构交互式查看 |

## 许可

MIT License
