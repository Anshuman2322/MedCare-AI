"""Dataset and DataLoader helpers for MedCare-AI."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    """Return image transform used for model training."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_inference_transform() -> transforms.Compose:
    """Return image transform used for validation, testing, and prediction."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Build train/val/test dataloaders from folder structure.

    Expected structure:
      data_dir/
        train/NORMAL, train/PNEUMONIA
        val/NORMAL, val/PNEUMONIA
        test/NORMAL, test/PNEUMONIA
    """
    data_dir = Path(data_dir)

    train_dataset = datasets.ImageFolder(
        root=str(data_dir / "train"),
        transform=get_train_transform(),
    )
    val_dataset = datasets.ImageFolder(
        root=str(data_dir / "val"),
        transform=get_inference_transform(),
    )
    test_dataset = datasets.ImageFolder(
        root=str(data_dir / "test"),
        transform=get_inference_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names


def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Backward-compatible alias for older code paths."""
    return get_dataloaders(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
