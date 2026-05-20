from __future__ import annotations

import argparse
import json
import importlib
import warnings
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter

from .data import build_dataset, decode, encode, get_device, load_text, seed_everything
from .models import build_model
from .training import load_model, save_model, train_model


warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
    category=Warning,
)

def _vocab_path(model_path: str) -> Path:
    return Path(f"{model_path}.vocab.json")


def _save_vocab(chars: list[str], model_path: str) -> None:
    path = _vocab_path(model_path)
    path.write_text(json.dumps({"chars": chars}, ensure_ascii=False), encoding="utf-8")


def _load_vocab(model_path: str) -> list[str] | None:
    path = _vocab_path(model_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    chars = payload.get("chars")
    if not isinstance(chars, list) or not chars:
        return None
    return chars


def _save_loss_curve(loss_history: list[tuple[int, float]], args: argparse.Namespace) -> Path:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save a loss curve") from exc

    output_dir = Path(args.loss_curve_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"loss_curve_{args.model}_{timestamp}.png"

    steps = [step for step, _ in loss_history]
    losses = [loss for _, loss in loss_history]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(steps, losses, color="#1f77b4", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve - model {args.model}")
    ax.grid(True, alpha=0.25)
    # logarimthmic scale
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())

    config_lines = [
        f"model: {args.model}",
        f"batch_size: {args.batch_size}",
        f"block_size: {args.block_size}",
        f"learning_rate: {args.learning_rate}",
        f"embed_dim: {args.embed_dim}",
        f"head_num: {args.head_num}",
        f"head_size: {args.head_size}",
        f"num_layers: {args.num_layers}",
        f"steps: {args.steps}",
        f"seed: {args.seed}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(config_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser(default_model: str = "v1") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and sample a character-level transformer")
    parser.add_argument("--model", "--variant", dest="model", default=default_model, choices=("v1", "v2", "v3"), help="model to train")
    parser.add_argument("--source", default=None, help="text file path or 'tinystories'")
    parser.add_argument("--limit", type=int, default=1000, help="number of TinyStories rows to use")
    parser.add_argument("--model-path", default="/model/model.pt", help="checkpoint path")
    parser.add_argument("--steps", type=int, default=5000, help="training steps")
    parser.add_argument("--log-interval", type=int, default=1000, help="log interval for printing loss")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--block-size", type=int, default=128, help="context window")
    parser.add_argument("--embed-dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--head-num", type=int, default=8, help="number of attention heads for v2/v3")
    parser.add_argument("--head-size", type=int, default=None, help="attention head size for v2/v3")
    parser.add_argument("--num-layers", type=int, default=4, help="number of transformer blocks for v3")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="optimizer learning rate")
    parser.add_argument("--generate-tokens", type=int, default=200, help="tokens to sample after training")
    parser.add_argument("--start-text", default="Once upon a time", help="prompt used for sampling")
    parser.add_argument("--device", default=None, help="override device, for example cpu or cuda")
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--loss-curve-dir", default=None, help="folder for saving a loss curve image after training")
    parser.add_argument("--load-model", action="store_true", help="load from model path or train a new model")
    parser.add_argument("--eval", action="store_true", help="eval model")
    parser.add_argument("--train", action="store_true", help="train model")
    return parser


def main(default_model: str = "v1") -> None:
    args = build_parser(default_model=default_model).parse_args()
    seed_everything(args.seed)

    device = get_device(args.device)
    print(f"using device: {device}")

    dataset = None
    chars = None

    if args.train:
        text = load_text(args.source, limit=args.limit)
        dataset = build_dataset(text)
        chars = dataset.chars
    elif args.eval:
        chars = _load_vocab(args.model_path)
        if chars is None:
            print("vocab metadata missing; rebuilding dataset for eval fallback")
            text = load_text(args.source, limit=args.limit)
            dataset = build_dataset(text)
            chars = dataset.chars

    if chars is None:
        raise ValueError("no vocabulary available: run with --train first or use --source to rebuild")

    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}

    model = build_model(
        variant=args.model,
        vocab_size=len(chars),
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        head_num=args.head_num,
        head_size=args.head_size,
        num_layers=args.num_layers,
    ).to(device)

    if args.load_model or args.eval:
        model = load_model(model, args.model_path, device)
        print(f"loaded model from {args.model_path}")

    if args.train:
        assert dataset is not None
        loss_history = train_model(
            model,
            dataset.data,
            steps=args.steps,
            log_interval=args.log_interval,
            batch_size=args.batch_size,
            block_size=args.block_size,
            lr=args.learning_rate,
            device=device,
        )

        save_model(model, args.model_path)
        print(f"saved model to {args.model_path}")
        _save_vocab(chars, args.model_path)
        print(f"saved vocab metadata to {_vocab_path(args.model_path)}")

        if args.loss_curve_dir:
            if loss_history:
                curve_path = _save_loss_curve(loss_history, args)
                print(f"saved loss curve to {curve_path}")
            else:
                print("no loss values were recorded; skipping loss curve export")

    if args.eval:
        model.eval()

        fallback_char = chars[0]
        start_tokens = encode(args.start_text, stoi, fallback_char=fallback_char).unsqueeze(0).to(device)
        generated = model.generate(start_tokens, max_new_tokens=args.generate_tokens)
        print(decode(generated[0].cpu(), itos))
