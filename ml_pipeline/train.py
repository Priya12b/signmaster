import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ml_pipeline.model import get_asl_model
from ml_pipeline.dataset import prepare_asl_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def get_transforms(img_size=128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform

def train_model(
    epochs=4,
    batch_size=64,
    lr=0.0004,
    img_size=128,
    architecture="mobilenet_v2",
    resume=True,
    device=None
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples across {num_classes} classes.", flush=True)

    # Instantiate model
    model = get_asl_model(num_classes=num_classes, pretrained=False, architecture=architecture)
    best_model_path = os.path.join(MODELS_DIR, "asl_mobilenetv2_best.pth")
    best_val_acc = 0.0

    if resume and os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from checkpoint with prior Best Val Acc: {best_val_acc*100:.2f}%", flush=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n--- Starting Fine-Tuning ---", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%",
            flush=True
        )

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "class_names": class_names,
                "architecture": architecture,
                "img_size": img_size
            }, best_model_path)
            print(f" -> Saved new best model (Val Acc: {best_val_acc*100:.2f}%) to {best_model_path}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nFine-tuning completed in {elapsed:.1f} seconds. Best Validation Accuracy: {best_val_acc*100:.2f}%", flush=True)
    return model, best_val_acc

if __name__ == "__main__":
    train_model(epochs=3)
