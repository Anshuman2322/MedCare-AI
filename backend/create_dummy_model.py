"""Create a randomly initialized checkpoint for local API testing."""

from model.model import build_model, save_model_weights


def main() -> None:
    """Save untrained model weights to medcare_model.pth."""
    model = build_model(device="cpu")
    save_model_weights(model, "medcare_model.pth")
    print("Dummy model saved successfully as medcare_model.pth")


if __name__ == "__main__":
    main()
