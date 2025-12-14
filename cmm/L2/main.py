from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from metrics import format_confusion_matrix
from model import MnistCnn
from predict import predict_image, predict_random_samples
from train_eval import train_one_epoch, evaluate
from utils import resolve_device, set_seed


def build_datasets(data_dir: Path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
        ]
    )
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    return train_ds, test_ds


def cmd_train(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = resolve_device(force_cpu=args.cpu)

    data_dir = Path(args.data_dir)
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = build_datasets(data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MnistCnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    ckpt_path = artifacts / args.checkpoint_name

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        ev, cm = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs}: "
              f"train loss={tr.loss:.4f} acc={tr.acc:.4f} | "
              f"test loss={ev.loss:.4f} acc={ev.acc:.4f}")

        if ev.acc > best_acc:
            best_acc = ev.acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": ev.acc,
                    "args": vars(args),
                },
                ckpt_path,
            )

    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Saved checkpoint: {ckpt_path}")

    # show a few predictions
    samples = predict_random_samples(model, test_ds, device, samples=args.samples, seed=args.seed)
    for i, (true_y, pred_y, probs) in enumerate(samples, 1):
        topv, topi = probs.topk(3)
        top3 = ", ".join(f"{int(idx)}:{float(val):.3f}" for idx, val in zip(topi, topv))
        print(f"Sample {i}: true={true_y} pred={pred_y} top3={top3}")

    return 0


def _load_model(checkpoint: Path, device: torch.device) -> MnistCnn:
    ckpt = torch.load(checkpoint, map_location=device)
    model = MnistCnn().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def cmd_eval(args: argparse.Namespace) -> int:
    device = resolve_device(force_cpu=args.cpu)

    data_dir = Path(args.data_dir)
    _, test_ds = build_datasets(data_dir)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _load_model(Path(args.checkpoint), device)
    criterion = nn.CrossEntropyLoss()

    ev, cm = evaluate(model, test_loader, criterion, device)
    print(f"Eval: loss={ev.loss:.4f} acc={ev.acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(format_confusion_matrix(cm))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    device = resolve_device(force_cpu=args.cpu)
    model = _load_model(Path(args.checkpoint), device)

    if args.image:
        pred, probs = predict_image(model, args.image, device)
        topv, topi = probs.topk(5)
        top5 = ", ".join(f"{int(idx)}:{float(val):.3f}" for idx, val in zip(topi, topv))
        print(f"Image: {args.image}")
        print(f"Predicted: {pred}")
        print(f"Top5: {top5}")
        return 0

    data_dir = Path(args.data_dir)
    _, test_ds = build_datasets(data_dir)
    samples = predict_random_samples(model, test_ds, device, samples=args.samples, seed=args.seed)
    for i, (true_y, pred_y, probs) in enumerate(samples, 1):
        topv, topi = probs.topk(3)
        top3 = ", ".join(f"{int(idx)}:{float(val):.3f}" for idx, val in zip(topi, topv))
        print(f"Sample {i}: true={true_y} pred={pred_y} top3={top3}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab2", description="Lab 2: MNIST object recognition with a CNN (PyTorch)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--data-dir", default="data", help="Folder for downloading MNIST")
        sp.add_argument("--artifacts-dir", default="artifacts", help="Folder for checkpoints/artifacts")
        sp.add_argument("--batch-size", type=int, default=128)
        sp.add_argument("--workers", type=int, default=2)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")

    t = sub.add_parser("train", help="Train CNN on MNIST")
    add_common(t)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--checkpoint-name", default="mnist_cnn.pt")
    t.add_argument("--samples", type=int, default=5, help="How many random test predictions to print after training")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="Evaluate a checkpoint")
    add_common(e)
    e.add_argument("--checkpoint", required=True)
    e.set_defaults(func=cmd_eval)

    pr = sub.add_parser("predict", help="Predict digits (random samples or an image)")
    add_common(pr)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--samples", type=int, default=8)
    pr.add_argument("--image", default=None, help="Path to an image (png/jpg) with a digit")
    pr.set_defaults(func=cmd_predict)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
