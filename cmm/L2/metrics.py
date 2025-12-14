from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(pred_logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = pred_logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    # preds/targets: shape [N], dtype long
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def format_confusion_matrix(cm: torch.Tensor) -> str:
    # rows = true, cols = predicted
    n = cm.size(0)
    header = "     " + " ".join(f"{i:>5d}" for i in range(n))
    lines = [header]
    for i in range(n):
        row = " ".join(f"{cm[i, j].item():>5d}" for j in range(n))
        lines.append(f"{i:>3d} | {row}")
    return "\n".join(lines)
