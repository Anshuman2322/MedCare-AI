"""Prediction utilities and CLI for MedCareCNN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
import torch

from .data_loader import get_inference_transform
from .model import load_model_weights

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_prediction_model(model_path: str, device: torch.device):
    """Load trained model weights for prediction."""
    return load_model_weights(model_path=model_path, device=device)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image into a model input tensor."""
    transform = get_inference_transform()
    return transform(image).unsqueeze(0)


def predict_tensor(model, image_tensor: torch.Tensor, device: torch.device):
    """Predict class and confidence for a preprocessed tensor."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    label = CLASS_NAMES[predicted_idx.item()]
    return label, float(confidence.item())


def predict_image(model, image_path: str, device: torch.device):
    """Predict one image path and return a result dictionary."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    label, confidence = predict_tensor(model, image_tensor, device)

    return {
        "image": image_path,
        "prediction": label,
        "confidence": confidence,
    }


def collect_images_from_folder(folder_path: str):
    """Collect image files recursively from a folder."""
    valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    folder = Path(folder_path)

    image_paths = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in valid_suffixes
    ]
    image_paths.sort()
    return image_paths


def save_predictions_csv(results, out_csv: str) -> None:
    """Save a list of prediction dictionaries to CSV."""
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["image", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for CLI prediction."""
    parser = argparse.ArgumentParser(description="Predict chest X-ray class from image(s)")
    parser.add_argument("--model", type=str, default="medcare_model.pth", help="Path to model .pth file")
    parser.add_argument("--out_csv", type=str, default="outputs/predictions.csv", help="CSV output for folder mode")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Single image path")
    source_group.add_argument("--folder", type=str, help="Folder path (recursive)")
    return parser.parse_args()


def main() -> None:
    """Run CLI prediction for one image or a full folder."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.model).exists():
        print(f"Model file not found: {args.model}")
        return

    model = load_prediction_model(model_path=args.model, device=device)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {args.image}")
            return

        result = predict_image(model, str(image_path), device)
        print(f"Image: {result['image']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        return

    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Folder not found: {args.folder}")
        return

    image_paths = collect_images_from_folder(str(folder_path))
    if not image_paths:
        print("No images found in folder.")
        return

    results = []
    for path in image_paths:
        result = predict_image(model, str(path), device)
        results.append(result)
        print(f"{path.name} -> {result['prediction']} ({result['confidence']:.3f})")

    save_predictions_csv(results, args.out_csv)
    print(f"Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
