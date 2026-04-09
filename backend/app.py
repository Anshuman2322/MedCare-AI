"""Flask API for MedCare-AI prediction."""

from __future__ import annotations

import io

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import torch

from model.predict import load_prediction_model, predict_tensor, preprocess_image

MODEL_PATH = "medcare_model.pth"
DEVICE = torch.device("cpu")

app = Flask(__name__)
CORS(app)

# Load once at startup so each request only runs inference.
model = load_prediction_model(model_path=MODEL_PATH, device=DEVICE)


@app.route("/")
def home():
    """Health-check endpoint."""
    return jsonify({"message": "MedCare-AI Flask Backend is Running!"})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """Predict image class from multipart form upload key named 'file'."""
    if "file" not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Invalid image: {exc}"}), 400

    image_tensor = preprocess_image(image)
    label, confidence = predict_tensor(model, image_tensor, DEVICE)

    return jsonify(
        {
            "result": "Healthy" if label == "NORMAL" else "Pneumonia",
            "confidence": round(confidence, 4),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
