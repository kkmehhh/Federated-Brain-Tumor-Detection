import argparse
import torch
import torch.nn as nn
import flwr as fl
import os
import random
from data_utils import get_loaders
try:
    from models.xray_model import load_xray_model
except ImportError:
    from xray_model import load_xray_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LIMIT_IMAGES = 50  # RULE: Only train on 50 images per round

class BrainClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, client_id):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def _evaluate_accuracy(self, name):
        """Helper to check accuracy instantly."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total if total > 0 else 0
        print(f"   {name} Accuracy: {acc:.2%}")
        return acc

    def fit(self, parameters, config):
        # 1. Load Global Weights (The "Before" State)
        self.set_parameters(parameters)
        
        print(f"\n[CLIENT {self.client_id}] Starting Round...")
        
        # 2. Evaluate Global Model (Before Training)
        print(" Checking Global Model performance...")
        self._evaluate_accuracy("Global Model (Before Training)")

        # 3. Train on LIMITED Data (50 images)
        self.model.train()
        images_processed = 0
        
        print(f" Training on max {LIMIT_IMAGES} images...")
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Check limit
            if images_processed >= LIMIT_IMAGES:
                break
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            images_processed += len(labels)

        # 4. Evaluate Local Model (After Training)
        print(" Checking Local Model performance...")
        self._evaluate_accuracy("Local Model (After Training)")

        # 5. SAVE LOCAL MODEL
        local_filename = f"client_{self.client_id}_trained.pth"
        torch.save(self.model.state_dict(), local_filename)
        print(f" Saved local model to: {local_filename}")

        return self.get_parameters(config={}), images_processed, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = 0.0
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += self.loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return float(loss), total, {"accuracy": accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to client data folder")
    parser.add_argument("--id", type=str, default="1", help="Client ID for saving files")
    args = parser.parse_args()

    # Load Model (Try to load previous global model if exists, else random)
    start_weights = "global_model.pth" if os.path.exists("global_model.pth") else None
    model = load_xray_model(DEVICE, weights_path=start_weights)

    train_loader, val_loader = get_loaders(args.data, batch_size=10)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BrainClient(model, train_loader, val_loader, args.id)
    )