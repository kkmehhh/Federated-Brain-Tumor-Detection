import argparse, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.xray_model import load_xray_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(data, weights="models/pretrained_xray.pth"):
    model = load_xray_model(device=DEVICE, weights_path=weights)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor()
    ])

    ds = datasets.ImageFolder(data, transform=transform)
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    correct, total = 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_sum = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += loss_fn(out, y).item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)

    print("Before FL:")
    print("Accuracy:", correct/total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    evaluate(args.data)
