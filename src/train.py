import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model import SASRec


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
    model = SASRec(
        num_items=num_items,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_hit = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model.pt"

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
        print(
            f"epoch={epoch} loss={avg_loss:.4f} "
            f"val_hit@10={val_metrics['hit@10']:.4f} val_ndcg@10={val_metrics['ndcg@10']:.4f}"
        )

        if val_metrics["hit@10"] >= best_hit:
            best_hit = val_metrics["hit@10"]
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
    model.load_state_dict(state["model_state"])
    test_metrics = evaluate(model, test_samples, num_items, device, args.max_len, k=10)
    print(f"test_hit@10={test_metrics['hit@10']:.4f} test_ndcg@10={test_metrics['ndcg@10']:.4f}")
    print(f"checkpoint saved to: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Train Transformer sequential recommender (SASRec)")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
