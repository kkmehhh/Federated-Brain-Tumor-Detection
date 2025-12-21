import torch.nn as nn
from torchvision.models import resnet50


def build_resnet50(num_classes=3):
    model = resnet50(weights=None)   # Ensure same architecture
    model.fc = nn.Linear(2048, num_classes)
    return model
