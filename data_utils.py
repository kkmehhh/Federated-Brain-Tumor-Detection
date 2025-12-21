import os
import sys
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

IMG_SIZE = 224

# Preprocessing: Resize and Normalize
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def get_loaders(data_root, batch_size=10):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.exists(train_dir):
        print(f"❌ Error: {train_dir} does not exist!")
        sys.exit(1)

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    # Batch size 10 makes it easy to count "50 images" (5 batches)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader