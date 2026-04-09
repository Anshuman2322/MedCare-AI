"""Train MedCareCNN on chest X-ray image folders.

Example:
    python -m model.train --data_dir data --epochs 5
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from .data_loader import get_dataloaders
from .model import build_model, save_model_weights


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    """Evaluate a model on a dataloader and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    data_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    model_path: str = "medcare_model.pth",
    num_workers: int = 0,
) -> None:
    """Run a full train/val/test workflow and save best weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Classes: {class_names}")

    model = build_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_weights(model, model_path)
            print(f"Saved best model to: {model_path}")

    # Load best model for final test report.
    best_model = build_model(device=device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MedCareCNN")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--model_path", type=str, default="medcare_model.pth", help="Output .pth path")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    """Entry point for command line training."""
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
