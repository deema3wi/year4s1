from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def _mnist_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
        ]
    )


@torch.no_grad()
def predict_image(model: torch.nn.Module, image_path: str | Path, device: torch.device) -> tuple[int, torch.Tensor]:
    model.eval()
    p = Path(image_path)
    img = Image.open(p).convert("RGB")
    x = _mnist_preprocess()(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred = int(probs.argmax().item())
    return pred, probs


@torch.no_grad()
def predict_random_samples(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    samples: int = 8,
    seed: int = 1,
) -> list[tuple[int, int, torch.Tensor]]:
    """Returns list of (true, pred, probs)."""
    model.eval()
    rng = random.Random(seed)
    idxs = [rng.randrange(0, len(dataset)) for _ in range(samples)]
    out = []
    for i in idxs:
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        pred = int(probs.argmax().item())
        out.append((int(y), pred, probs))
    return out
