import argparse
import json
from pathlib import Path

import torch

from src.model import SASRec


@torch.no_grad()
def recommend(
    dataset_file: Path,
    checkpoint_file: Path,
    user: str,
    topk: int = 10,
    max_len: int = 50,
):
    with dataset_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    state = torch.load(checkpoint_file, map_location="cpu")
    config = state["config"]

    model = SASRec(
        num_items=config["num_items"],
        max_len=config["max_len"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )
    model.load_state_dict(state["model_state"])
    model.eval()

    history = data["user_histories"].get(user)
    if not history:
        raise ValueError(
            f"User '{user}' not found in dataset. Please check keys in user_histories."
        )

    history = history[-max_len:]
    input_ids = torch.tensor([[0] * (max_len - len(history)) + history], dtype=torch.long)
    scores = model(input_ids)[0]
    scores[0] = float("-inf")

    for item_id in set(history):
        scores[item_id] = float("-inf")

    num_items = config["num_items"]
    item_ids = torch.topk(scores[1 : num_items + 1], k=min(topk, num_items)).indices + 1
    recs = []
    for idx in item_ids.tolist():
        movie_id = data["id2item"][str(idx)]
        title = data["id2title"][str(idx)]
        recs.append({"movie_id": movie_id, "title": title})
    return recs


def main():
    parser = argparse.ArgumentParser(description="Generate movie recommendations for a user")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--user", required=True, type=str)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=50)
    args = parser.parse_args()

    recs = recommend(args.dataset, args.checkpoint, args.user, args.topk, args.max_len)
    print(json.dumps({"user": args.user, "recommendations": recs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
