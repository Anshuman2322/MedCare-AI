"""Model definition and model utility helpers.

This file keeps all CNN architecture related logic in one place so training,
prediction, and the Flask app can share the exact same model code.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedCareCNN(nn.Module):
    """A small CNN for binary chest X-ray classification.

    The output has two logits in class order:
    0 -> NORMAL
    1 -> PNEUMONIA
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        # Feature extractor
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Classifier for 224x224 input images
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Args:
            x: Input image batch with shape (N, 3, 224, 224).

        Returns:
            Logits tensor with shape (N, 2).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(device: torch.device | str = "cpu") -> MedCareCNN:
    """Create a new model and move it to the requested device."""
    model = MedCareCNN()
    return model.to(device)


def save_model_weights(model: nn.Module, model_path: str | Path) -> None:
    """Save model weights to a .pth file."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_model_weights(
    model_path: str | Path,
    device: torch.device | str = "cpu",
) -> MedCareCNN:
    """Load model weights and return a model ready for inference.

    Supports plain state dict files and dicts containing 'model_state'.
    """
    model = build_model(device=device)
    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model
