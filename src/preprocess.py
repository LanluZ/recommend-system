import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {"电影id", "标题", "用户名"}


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required columns: {REQUIRED_COLUMNS}")
        for row in reader:
            movie_id = str(row["电影id"]).strip()
            user = str(row["用户名"]).strip()
            title = str(row["标题"]).strip()
            if not movie_id or not user or not title:
                continue
            yield user, movie_id, title


def build_datasets(input_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    user_histories = defaultdict(list)
    movie_titles = {}
    user_item_seen = defaultdict(set)

    for user, movie_id, title in _read_rows(input_csv):
        if movie_id in user_item_seen[user]:
            continue
        user_item_seen[user].add(movie_id)
        user_histories[user].append(movie_id)
        movie_titles[movie_id] = title

    item2id = {"<PAD>": 0}
    id2item = {"0": "<PAD>"}
    id2title = {"0": "<PAD>"}
    for movie_id, title in movie_titles.items():
        idx = len(item2id)
        item2id[movie_id] = idx
        id2item[str(idx)] = movie_id
        id2title[str(idx)] = title

    user_histories_id = {
        user: [item2id[movie_id] for movie_id in seq]
        for user, seq in user_histories.items()
        if len(seq) >= 2
    }

    train_samples = []
    val_samples = []
    test_samples = []

    for user, seq in user_histories_id.items():
        n = len(seq)
        train_cut = n - 2 if n >= 3 else n - 1

        for t in range(1, train_cut):
            train_samples.append({"user": user, "history": seq[:t], "target": seq[t]})

        if not any(s["user"] == user for s in train_samples):
            train_samples.append({"user": user, "history": seq[:1], "target": seq[1]})

        if n >= 3:
            val_samples.append({"user": user, "history": seq[:-2], "target": seq[-2]})
            test_samples.append({"user": user, "history": seq[:-1], "target": seq[-1]})
        else:
            test_samples.append({"user": user, "history": seq[:-1], "target": seq[-1]})

    payload = {
        "meta": {
            "num_items": len(item2id) - 1,
            "num_users": len(user_histories_id),
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "num_test_samples": len(test_samples),
        },
        "item2id": item2id,
        "id2item": id2item,
        "id2title": id2title,
        "user_histories": user_histories_id,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }

    output_file = output_dir / "dataset.json"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))
    print(f"processed dataset saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate and preprocess movie recommendation data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/clean.csv"),
        help="Path to raw CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data",
    )
    args = parser.parse_args()
    build_datasets(args.input, args.output_dir)


if __name__ == "__main__":
    main()
