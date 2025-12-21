# Federated Learning for Multi-Class Brain Tumor Differential Diagnosis

A privacy-preserving AI system that enables multiple institutions to collaboratively train a deep learning model for brain tumor classification without sharing sensitive patient data.

## Project Overview

Standard medical AI requires aggregating patient data into a central server, which raises significant privacy and HIPAA compliance concerns. This system solves that problem using **Federated Learning**, where the model travels to the data, rather than the data traveling to the model.

The system performs differential diagnosis on MRI scans to classify four specific conditions:
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor**

## Key Features

* **Privacy-Preserving Architecture:** Raw patient MRI scans never leave the local client (hospital). Only model weight updates (gradients) are transmitted to the central server.
* **Differential Diagnosis:** Unlike simple binary classifiers (Tumor vs. No Tumor), this model distinguishes between specific tumor types, providing higher clinical utility.
* **Byzantine Resilience (KRUM):** The server implements the **KRUM aggregation algorithm** to detect and reject malicious or corrupted updates, ensuring the global model remains accurate even if a client malfunctions.
* **Independent Server Evaluation:** The central server maintains a separate, unseen test dataset to independently validate the global model's performance after every training round.

## Codebase & File Descriptions

Here is a detailed breakdown of the project files and their specific roles:

### 1. Core Logic
* **`fl_server.py` (The Central Brain):**
  * **Role:** Orchestrates the entire federated learning process.
  * **Key Function:** Implements the custom `KRUMFedAvg` strategy class. It receives weight updates from clients, filters out "bad" updates using the KRUM distance metric, aggregates the good ones, and updates the global model.
  * **Evaluation:** Runs a standalone validation loop on the server's test set to track global accuracy.

* **`fl_client.py` (The Hospital Node):**
  * **Role:** Represents a single hospital or institution.
  * **Key Function:** Connects to the server, receives the latest global model, trains it locally on private data (e.g., `client1_data`), and computes the new weights.
  * **Features:** Includes a safety mechanism to save the local model (`client_x.pth`) before sending updates, ensuring the hospital retains its own specialized version.

* **`models/xray_model.py` (The Neural Network):**
  * **Role:** Defines the Deep Learning architecture.
  * **Key Function:** Uses a **ResNet18** backbone pre-trained on ImageNet. It replaces the final fully connected layer with a custom layer outputting 4 logits (one for each tumor class).

### 2. Utilities
* **`split_clients.py`:**
  * **Role:** Data preparation script.
  * **Key Function:** Takes the downloaded Kaggle dataset and splits it into simulated "hospital" folders (`client1_data`, `client2_data`, etc.) to create a realistic federated environment.

* **`data_utils.py`:**
  * **Role:** Data loading helper.
  * **Key Function:** handles image preprocessing (resizing, normalization) and creates PyTorch `DataLoader` objects for the training loops.
## How to Run

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**
    Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) and run the splitter script:
    ```bash
    python split_clients.py
    ```

3.  **Start the Server**
    ```bash
    python fl_server.py
    ```

4.  **Start Clients (Open separate terminals)**
    ```bash
    python fl_client.py --data client1_data --id 1
    python fl_client.py --data client2_data --id 2
    ```


make it better and specify which file does what
