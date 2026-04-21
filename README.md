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
outputs/
  model.pt
```

## 1) 安装依赖

```bash
pip install -r requirements.txt
```

## 2) 预处理数据（检查字段并构建训练集）

```bash
python -m src.preprocess --input data/raw/clean.csv --output-dir data/processed
```

## 3) 训练模型

```bash
python -m src.train --dataset data/processed/dataset.json --output-dir outputs --max-len 50 --weight-decay 1e-4
```

## 4) 生成推荐

```bash
python -m src.recommend --dataset data/processed/dataset.json --checkpoint outputs/model.pt --user "Kaito" --topk 10
```

## 方法说明

使用行业常见的序列推荐方法 SASRec：
- `Transformer Encoder` 建模用户最近行为序列；
- 因果 Mask 保证只利用历史行为预测下一个电影；
- 采用 Top-K 指标（Hit@10 / NDCG@10）进行验证。
