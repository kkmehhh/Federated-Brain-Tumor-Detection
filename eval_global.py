# eval_global.py
import torch
import argparse
from model import build_resnet50
from data_utils import get_loaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    model = build_resnet50()
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    _, val_loader, classes = get_loaders(args.data)
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    print("Test accuracy:", correct / total)
