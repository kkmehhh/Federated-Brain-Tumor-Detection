# split_data_multi_client.py
import os
import shutil
import random

# CONFIGURATION
SOURCE_DIR = "source_data"   # Your raw Kaggle download folder
CLIENTS = ["client1_data", "client2_data"] # We will make 2 separate datasets
SPLIT_RATIO = 0.8            # 80% Train, 20% Val

def split_dataset():
    # 1. Get class names
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f"Found classes: {classes}")

    # 2. Loop through each class (e.g., 'glioma', 'pituitary')
    for class_name in classes:
        src_path = os.path.join(SOURCE_DIR, class_name)
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # Split images broadly between Client 1 and Client 2 (50/50)
        mid_point = len(images) // 2
        client_chunks = {
            "client1_data": images[:mid_point],
            "client2_data": images[mid_point:]
        }

        # 3. Process each client
        for client_folder, client_images in client_chunks.items():
            # Create train/val folders for this client
            train_dir = os.path.join(client_folder, "train", class_name)
            val_dir = os.path.join(client_folder, "val", class_name)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Split this client's chunk into Train (80%) and Val (20%)
            split_idx = int(len(client_images) * SPLIT_RATIO)
            train_imgs = client_images[:split_idx]
            val_imgs = client_images[split_idx:]

            # Copy files
            for img in train_imgs:
                shutil.copy(os.path.join(src_path, img), os.path.join(train_dir, img))
            for img in val_imgs:
                shutil.copy(os.path.join(src_path, img), os.path.join(val_dir, img))
            
            print(f"  -> {client_folder}/{class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

    print("\n✅ DONE! You now have 'client1_data' and 'client2_data'.")

if __name__ == "__main__":
    split_dataset()
