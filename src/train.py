import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import SASRec


class SequenceDataset(Dataset):
    def __init__(self, samples: List[Dict], max_len: int = 50):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        history = item["history"][-self.max_len :]
        return {
            "history": history,
            "target": item["target"],
        }


def collate_fn(batch, max_len: int):
    # 统一左侧补零到固定长度，保证批量张量形状一致。
    histories = []
    targets = []
    for row in batch:
        seq = row["history"]
        pad = [0] * (max_len - len(seq))
        histories.append(pad + seq)
        targets.append(row["target"])
    return {
        "input_ids": torch.tensor(histories, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


@torch.no_grad()
def evaluate(model, samples, num_items, device, max_len=50, k=10):
    if not samples:
        return {"hit@10": 0.0, "ndcg@10": 0.0}

    model.eval()
    hits, ndcgs = 0.0, 0.0
    for row in samples:
        seq = row["history"][-max_len:]
        pad = [0] * (max_len - len(seq))
        input_ids = torch.tensor([pad + seq], dtype=torch.long, device=device)
        scores = model(input_ids)[0]

        # 评估时过滤历史已看物品（保留目标物品）与 padding id。
        seen = set(seq)
        for item_id in seen:
            if item_id != row["target"]:
                scores[item_id] = float("-inf")
        scores[0] = float("-inf")

        topk = torch.topk(scores[1 : num_items + 1], k=min(k, num_items)).indices + 1
        topk = topk.tolist()
        if row["target"] in topk:
            hits += 1
            rank = topk.index(row["target"]) + 1
            ndcgs += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()

    n = len(samples)
    return {"hit@10": hits / n, "ndcg@10": ndcgs / n}


def export_onnx(model: nn.Module, num_items: int, max_len: int, onnx_path: Path, device: torch.device):
    # 使用固定形状 dummy input 导出 ONNX，batch 维度设置为动态。
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.zeros((1, max_len), dtype=torch.long, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def train(args):
    with args.dataset.open("r", encoding="utf-8") as file:
        data = json.load(file)

    train_samples = data["train_samples"]
    val_samples = data["val_samples"]
    test_samples = data["test_samples"]
    num_items = data["meta"]["num_items"]

    train_dataset = SequenceDataset(train_samples, max_len=args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, args.max_len),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SASRec 主模型，超参数来自命令行。
    model = SASRec(
        num_items=num_items,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_hit = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"
    onnx_file = args.output_dir / "model.onnx"
    train_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_samples, num_items, device, args.max_len, k=10)
        train_history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "val_hit@10": val_metrics["hit@10"],
                "val_ndcg@10": val_metrics["ndcg@10"],
            }
        )
        print(
            f"epoch={epoch} loss={avg_loss:.4f} "
            f"val_hit@10={val_metrics['hit@10']:.4f} val_ndcg@10={val_metrics['ndcg@10']:.4f}"
        )

        if val_metrics["hit@10"] >= best_val_hit:
            # 以验证集 hit@10 作为 early-best 策略保存最优权重。
            best_val_hit = val_metrics["hit@10"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "num_items": num_items,
                        "max_len": args.max_len,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "dropout": args.dropout,
                    },
                },
                checkpoint,
            )

    state = torch.load(checkpoint, map_location=device)
    # 回载最优权重，在测试集上评估并导出产物。
    model.load_state_dict(state["model_state"])
    test_metrics = evaluate(model, test_samples, num_items, device, args.max_len, k=10)

    export_onnx(model, num_items=num_items, max_len=args.max_len, onnx_path=onnx_file, device=device)

    params_payload = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "dataset_meta": data["meta"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "best_val_hit@10": best_val_hit,
        "test_metrics": test_metrics,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "onnx": str(onnx_file),
        },
        "history": train_history,
    }
    params_file = args.output_dir / "training_params.json"
    with params_file.open("w", encoding="utf-8") as file:
        json.dump(params_payload, file, ensure_ascii=False, indent=2)

    print(f"test_hit@10={test_metrics['hit@10']:.4f} test_ndcg@10={test_metrics['ndcg@10']:.4f}")
    print(f"checkpoint saved to: {checkpoint}")
    print(f"onnx model saved to: {onnx_file}")
    print(f"training params saved to: {params_file}")


def main():
    parser = argparse.ArgumentParser(description="Train Transformer sequential recommender (SASRec)")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
