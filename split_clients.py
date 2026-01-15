import os
import shutil
import random

# --- CONFIGURATION ---
# The folder where your current 'train' and 'val' folders are:
SOURCE_FOLDER = "client_data" 

# The new folders we will create:
CLIENTS = ["client1_data", "client2_data"] 

def split_data_simply():
    # 1. Check if source data exists
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: I can't find the folder '{SOURCE_FOLDER}'")
        print("Make sure you are running this script in the same folder as 'client_data'")
        return

    print("Found data. Splitting into Client 1 and Client 2...")

    # 2. Process 'train' and 'val' folders
    for split_type in ["train", "val"]:
        current_path = os.path.join(SOURCE_FOLDER, split_type)
        
        # Skip if folder missing
        if not os.path.exists(current_path): 
            continue

        # Get all the classes (e.g., glioma, no_tumor)
        classes = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]

        for class_name in classes:
            # Full path to the images
            class_dir = os.path.join(current_path, class_name)
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Shuffle so it's random
            random.shuffle(images)

            # Split 50/50
            mid_point = len(images) // 2
            datasets = {
                CLIENTS[0]: images[:mid_point], # First half to Client 1
                CLIENTS[1]: images[mid_point:]  # Second half to Client 2
            }

            # Copy the files
            for client_name, img_list in datasets.items():
                # Define destination: client1_data/train/glioma/
                dest_dir = os.path.join(client_name, split_type, class_name)
                os.makedirs(dest_dir, exist_ok=True)

                for img in img_list:
                    shutil.copy(os.path.join(class_dir, img), os.path.join(dest_dir, img))
            
            print(f"   Splitting {class_name} ({split_type}): {len(images)} images copied.")

    print("\n Done! You now have 'client1_data' and 'client2_data'.")

if __name__ == "__main__":
    split_data_simply()
