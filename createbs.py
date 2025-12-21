# init_model.py
import torch
import torch.nn as nn
from torchvision import models
import os

def create_seed_model():
    print("🧠 Initializing fresh ResNet18 for 4-Class Brain Tumor detection...")
    
    # 1. Download standard ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Modify for 4 classes (Glioma, Meningioma, Pituitary, No Tumor)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    
    # 3. Save it into the models folder
    os.makedirs("models", exist_ok=True)
    save_path = "models/pretrained_brain.pth"
    torch.save(model.state_dict(), save_path)
    
    print(f"✅ Success! Created '{save_path}'.")
    print("Both clients will load this file to start.")

if __name__ == "__main__":
    create_seed_model()
