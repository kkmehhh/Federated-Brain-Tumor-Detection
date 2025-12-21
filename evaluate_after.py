# evaluate_after.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.xray_model import load_xray_model   # FIXED IMPORT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(data_dir: str):
    # -----------------------
    #  Load global model
    # -----------------------
    if not os.path.exists("global_model.pth"):
        print("❌ global_model.pth NOT FOUND!")
        return

    print("Loading global model...")
    model = load_xray_model(device=DEVICE, weights_path="global_model.pth")
    model.eval()

    # -----------------------
    #  Data loader
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # -----------------------
    #  Run evaluation
    # -----------------------
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            _, predicted = outputs.max(1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    print("\nAfter FL:")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to validation folder")
    args = parser.parse_args()

    evaluate(args.data)