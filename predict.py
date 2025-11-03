# predict.py
import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from model import MedCareCNN

# --- Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def load_model(model_path, device):
    model = MedCareCNN()
    state = torch.load(model_path, map_location=device)
    # if saved state_dict only:
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None, f"ERROR opening image: {e}"

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
    classes = ['NORMAL', 'PNEUMONIA']  # same order as training ImageFolder
    return {"image": image_path, "prediction": classes[pred_idx], "confidence": confidence}, None

def collect_images_from_folder(folder_path):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    imgs = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(exts):
                imgs.append(os.path.join(root, f))
    imgs.sort()
    return imgs

def main():
    parser = argparse.ArgumentParser(description="Predict medical image(s) from any path")
    parser.add_argument("--model", type=str, default="medcare_model.pth", help="Path to trained model .pth")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to single image (absolute or relative)")
    group.add_argument("--folder", type=str, help="Path to folder containing images (will search recursively)")
    parser.add_argument("--out_csv", type=str, default="outputs/predictions.csv", help="Output CSV for folder predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    model = load_model(args.model, device)

    results = []
    if args.image:
        img_path = args.image
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return
        res, err = predict_image(model, img_path, device)
        if err:
            print(err)
        else:
            print(f"Image: {res['image']}\nPrediction: {res['prediction']}\nConfidence: {res['confidence']:.4f}")
    else:
        folder = args.folder
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            return
        imgs = collect_images_from_folder(folder)
        if len(imgs) == 0:
            print("No images found in folder.")
            return
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        # batch predict one-by-one (can be optimized)
        for p in imgs:
            res, err = predict_image(model, p, device)
            if err:
                print(f"{p} -> {err}")
            else:
                print(f"{os.path.basename(p)} -> {res['prediction']} ({res['confidence']:.3f})")
                results.append(res)
        # save CSV
        try:
            import csv
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "prediction", "confidence"])
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            print(f"\nSaved predictions to {args.out_csv}")
        except Exception as e:
            print("Could not save CSV:", e)

if __name__ == "__main__":
    main()
