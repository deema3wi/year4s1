from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import accuracy, confusion_matrix
from utils import AverageMeter


@dataclass(frozen=True)
class EpochResult:
    loss: float
    acc: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        loss_meter = loss_meter.update(loss.item(), batch_size)
        acc_meter = acc_meter.update(accuracy(logits, y), batch_size)

    return EpochResult(loss=loss_meter.avg, acc=acc_meter.avg)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[EpochResult, torch.Tensor]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)

        batch_size = x.size(0)
        loss_meter = loss_meter.update(loss.item(), batch_size)
        acc_meter = acc_meter.update((preds == y).float().mean().item(), batch_size)

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    cm = confusion_matrix(preds_cat, targets_cat, num_classes=num_classes)

    return EpochResult(loss=loss_meter.avg, acc=acc_meter.avg), cm
