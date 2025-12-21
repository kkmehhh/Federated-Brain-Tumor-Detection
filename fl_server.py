import flwr as fl
import numpy as np
import torch
import os
from typing import List, Tuple, Dict, Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- IMPORT FLOWER'S BUILT-IN CONVERTERS (THE FIX) ---
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# Import your model logic
try:
    from models.xray_model import load_xray_model
except ImportError:
    from xray_model import load_xray_model

DEVICE = torch.device("cpu") 
IMG_SIZE = 224

# --- SERVER-SIDE DATA LOADER ---
def get_server_eval_loader(data_root="./client_data"):
    val_dir = os.path.join(data_root, "val")
    if not os.path.exists(val_dir):
        print(f"⚠️ Server cannot find validation data at {val_dir}. Skipping server-side eval.")
        return None
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    return DataLoader(val_ds, batch_size=32, shuffle=False)

# --- HELPER: EVALUATE MODEL ---
def evaluate_current_model(weights: List[np.ndarray], loader: DataLoader) -> float:
    if loader is None: return 0.0

    model = load_xray_model(DEVICE)
    
    # Flower's converter gives us the correct shapes automatically now.
    # We just need to load them.
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total if total > 0 else 0.0

# --- KRUM LOGIC ---
def krum(updates: List[np.ndarray]) -> np.ndarray:
    n = len(updates)
    if n < 2: return updates[0]
    
    scores = []
    for i in range(n):
        dists = [np.linalg.norm(updates[i] - updates[j]) for j in range(n) if i != j]
        scores.append(sum(sorted(dists)[:max(1, n-2)]))
    
    winner_idx = np.argmin(scores)
    print(f" KRUM Algorithm chose Client #{winner_idx + 1} as the most reliable.")
    return updates[winner_idx]

# --- CUSTOM STRATEGY ---
class KRUMFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, eval_loader, **kwargs):
        super().__init__(**kwargs)
        self.eval_loader = eval_loader
        self.global_weights = None 

    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}
        
        print(f"\n======== ROUND {server_round} SERVER REPORT ========")

        # 0. Get Target Shapes (for Reconstruction)
        dummy_model = load_xray_model(DEVICE)
        target_shapes = [val.cpu().numpy().shape for val in dummy_model.state_dict().values()]

        # 1. EVALUATE BEFORE
        if self.global_weights is not None:
            acc_before = evaluate_current_model(self.global_weights, self.eval_loader)
            print(f"Server Accuracy BEFORE aggregation: {acc_before:.2%}")
        else:
            print(f"Server Accuracy BEFORE aggregation: 0.00% (First Round)")

        # 2. AGGREGATE (Apply KRUM)
        print(f"🔍 Aggregating updates from {len(results)} clients...")
        
        # USE BUILT-IN CONVERTER (Fixes the shape/format issues)
        client_params = [parameters_to_ndarrays(res.parameters) for _, res in results]
        
        # Flatten
        flat_updates = [np.concatenate([p.flatten() for p in c_params]) for c_params in client_params]
        
        # Select Winner
        selected_flat = krum(flat_updates)
        
        # Reconstruct
        rebuilt = []
        idx = 0
        for shape in target_shapes:
            size = int(np.prod(shape))
            layer_flat = selected_flat[idx:idx+size]
            rebuilt.append(layer_flat.reshape(shape))
            idx += size
        
        self.global_weights = rebuilt

        # 3. EVALUATE AFTER
        acc_after = evaluate_current_model(rebuilt, self.eval_loader)
        print(f"Server Accuracy AFTER  aggregation: {acc_after:.2%}")

        # 4. SAVE (As Dictionary)
        params_dict = zip(dummy_model.state_dict().keys(), rebuilt)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        try:
            torch.save(state_dict, "global_model.pth")
            print(f"Saved new global model to 'global_model.pth'")
        except Exception as e:
            print(f"Warning: Failed to save global model: {e}")

        print("======================================================\n")

        # USE BUILT-IN CONVERTER (Fixes the EOF error)
        return ndarrays_to_parameters(rebuilt), {}

# --- START SERVER ---
if __name__ == "__main__":
    print("Loading server-side validation data...")
    val_loader = get_server_eval_loader()

    strategy = KRUMFedAvg(
        eval_loader=val_loader,
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
    )
    
    print("Server Starting on 0.0.0.0:8080")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )