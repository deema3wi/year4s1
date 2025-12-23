import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import cv2
import numpy as np


def get_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_data_loaders(batch_size, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

def prepare_digit(img_chunk):
    """
    Приймає вирізаний шматок зображення (numpy array),
    додає рамку (padding), ресайзить до 28x28 і нормалізує для PyTorch.
    """
    h, w = img_chunk.shape
    if h > w:
        pad = (h - w) // 2
        img_chunk = cv2.copyMakeBorder(img_chunk, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (w - h) // 2
        img_chunk = cv2.copyMakeBorder(img_chunk, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    img_chunk = cv2.resize(img_chunk, (20, 20), interpolation=cv2.INTER_AREA)

    img_chunk = cv2.copyMakeBorder(img_chunk, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(img_chunk).unsqueeze(0)

def find_digits_on_image(image_path):
    """Шукає цифри на зображенні за допомогою OpenCV"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10: 
            digit_regions.append((x, y, w, h))

    digit_regions.sort(key=lambda r: r[0])

    cropped_tensors = []
    for x, y, w, h in digit_regions:
        roi = thresh[y:y+h, x:x+w]
        tensor = prepare_digit(roi)
        cropped_tensors.append(tensor)

    return cropped_tensors

def predict_local(model, img_dir, device):
    path = Path(img_dir)
    if not path.exists():
        print(f"[Увага] Папка {img_dir} не знайдена.")
        return

    images = [p for p in path.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    if not images:
        print(f"[Інфо] У папці {img_dir} немає зображень.")
        return

    model.eval()
    print(f"\n--- Розпізнавання (Сегментація ввімкнена) ---")
    
    with torch.no_grad():
        for img_path in images:
            digit_tensors = find_digits_on_image(img_path)
            
            if not digit_tensors:
                print(f"{img_path.name:<20} | Цифр не знайдено")
                continue

            full_number = ""
            confidences = []

            for tensor in digit_tensors:
                tensor = tensor.to(device)
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = probs.argmax().item()
                conf = probs[0][pred_idx].item()
                
                full_number += str(pred_idx)
                confidences.append(conf)

            avg_conf = (sum(confidences) / len(confidences)) * 100
            print(f"Файл: {img_path.name:<15} -> Число: {full_number:<10} (Сер. впевненість: {avg_conf:.1f}%)")

def run_training(epochs, batch_size, lr, device_name, save_path):
    device = get_device(force_cpu=(device_name == 'cpu'))
    print(f"Пристрій: {device}")
    train_loader, test_loader = get_data_loaders(batch_size)
    from model import MnistCnn
    model = MnistCnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Старт навчання ({epochs} епох)...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch} | Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Модель збережено.")

def run_prediction(checkpoint_path, img_dir, device_name):
    device = get_device(force_cpu=(device_name == 'cpu'))
    from model import MnistCnn
    model = MnistCnn().to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print("Модель не знайдена.")
        return
    predict_local(model, img_dir, device)
