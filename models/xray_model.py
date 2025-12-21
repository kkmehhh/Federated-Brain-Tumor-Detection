import torch
import torch.nn as nn
from torchvision import models

def load_xray_model(device, weights_path=None):
    # 1. Load standard ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Modify for 4 classes (Glioma, Meningioma, Pituitary, No Tumor)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4) 
    
    # 3. Load weights if provided
    if weights_path:
        try:
            # strict=False allows loading slightly different architectures safely
            model.load_state_dict(torch.load(weights_path), strict=False)
        except Exception as e:
            print(f"⚠️ Warning: Could not load weights from {weights_path}. Starting fresh. ({e})")
    
    return model.to(device)
