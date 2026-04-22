import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import netron
import torch

from model import SASRec


def write_loss_png(history: List[dict], output_file: Path):
    # 从训练历史中提取每轮 loss 并用 matplotlib 绘制折线图。
    losses = [float(row["loss"]) for row in history if "loss" in row]
    epochs = [int(row["epoch"]) for row in history if "epoch" in row]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, losses, marker="o", linewidth=2.2, markersize=6, color="#d9480f", label="Training Loss")
    ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax.set_title("Training Loss Curve", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if losses:
        min_loss = min(losses)
        max_loss = max(losses)
        last_loss = losses[-1]
        ax.text(
            0.98,
            0.97,
            f"epochs: {len(epochs)}\nmin loss: {min_loss:.4f}\nlast loss: {last_loss:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="#fff7ed", edgecolor="#fed7aa", linewidth=1.5),
        )

    fig.tight_layout()
    fig.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_model_text(config: dict, output_file: Path):
    # 生成可读的模型结构摘要，便于快速查看关键超参数。
    text = "\n".join(
        [
            "SASRec model summary",
            "====================",
            f"max_len: {config['max_len']}",
            f"hidden_size: {config['hidden_size']}",
            f"num_layers: {config['num_layers']}",
            f"num_heads: {config['num_heads']}",
            f"dropout: {config['dropout']}",
            f"ffn_dim: {config['hidden_size'] * 4}",
            "",
            "Flow:",
            "Input IDs -> Item/Position Embedding -> LayerNorm+Dropout ->",
            "Transformer Encoder Stack -> Last Valid Hidden -> Linear -> Logits",
        ]
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")


def export_torchscript_from_checkpoint(checkpoint_path: Path, scripted_path: Path) -> Path:
    # 从训练 checkpoint 重建模型并导出 TorchScript，供 Netron 可视化。
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    config = state.get("config", {})
    model_state = state.get("model_state")
    if model_state is None:
        raise ValueError("Checkpoint missing 'model_state'.")

    required = {"num_items", "max_len", "hidden_size", "num_layers", "num_heads", "dropout"}
    missing = sorted(required - set(config.keys()))
    if missing:
        raise ValueError(f"Checkpoint config missing keys: {missing}")

    model = SASRec(
        num_items=config["num_items"],
        max_len=config["max_len"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )
    model.load_state_dict(model_state)
    model.eval()

    try:
        # 优先 script，通常比 trace 更稳健。
        traced = torch.jit.script(model)
    except Exception:
        # script 失败时回退 trace，关闭一致性检查避免 Transformer 相关误报。
        dummy_input = torch.zeros((1, config["max_len"]), dtype=torch.long)
        traced = torch.jit.trace(model, dummy_input, check_trace=False)

    scripted_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(scripted_path))
    return scripted_path


def main():
    parser = argparse.ArgumentParser(description="Visualize model architecture and training loss")
    parser.add_argument("--training-params", type=Path, default=Path("outputs/training_params.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--scripted-model", type=Path, default=Path("outputs/model_netron.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--netron", action="store_true", help="Start Netron server for PT model visualization")
    parser.add_argument("--netron-host", type=str, default="127.0.0.1")
    parser.add_argument("--netron-port", type=int, default=8081)
    parser.add_argument("--browse", action="store_true", help="Open browser automatically when Netron starts")
    args = parser.parse_args()

    with args.training_params.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    history = payload.get("history", [])
    config = payload.get("hyperparameters", {})

    required_keys = {"max_len", "hidden_size", "num_layers", "num_heads", "dropout"}
    missing = sorted(required_keys - set(config.keys()))
    if missing:
        raise ValueError(f"Missing keys in hyperparameters: {missing}")

    loss_png = args.output_dir / "loss_curve.png"
    model_txt = args.output_dir / "model_summary.txt"

    write_loss_png(history, loss_png)
    write_model_text(config, model_txt)

    print(f"loss curve saved to: {loss_png}")
    print(f"model summary saved to: {model_txt}")

    if args.netron:
        # 启动 Netron 服务进行交互式结构查看。
        scripted_path = export_torchscript_from_checkpoint(args.checkpoint, args.scripted_model)
        address = (args.netron_host, args.netron_port)
        print(f"torchscript model saved to: {scripted_path}")
        print(f"starting netron at: http://{args.netron_host}:{args.netron_port}")
        print("press Ctrl+C to stop netron server")
        netron.start(file=str(scripted_path), address=address, browse=args.browse)
        input('键入以继续')


if __name__ == "__main__":
    main()
