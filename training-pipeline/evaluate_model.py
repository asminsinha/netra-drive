import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from train_model import NetraDriveDataset, MultiTaskNetraNet

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "driver_model.pth")

def run_evaluation_pipeline():
    print(f"Initializing Evaluation Engine on Device: [{DEVICE}]...")
    
    # Load the evaluation test dataset split
    test_dataset = NetraDriveDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    if len(test_dataset) == 0:
        print("[ERROR] No images found in the test split. Please verify your processed dataset directory structure.")
        return

    # Initialize model architecture and inject trained weights
    model = MultiTaskNetraNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Successfully loaded trained weights artifact: 'driver_model.pth'")
    else:
        print(f"[ERROR] Model weights file not found at {MODEL_PATH}")
        return
        
    model.eval()

    # Storage for predictions and ground truth labels per task
    task_metrics = {
        0: {"name": "EYES (Close/Open)", "preds": [], "targets": [], "target_names": ["Close", "Open"]},
        1: {"name": "YAWN (No Yawn/Yawn)", "preds": [], "targets": [], "target_names": ["No Yawn", "Yawn"]},
        2: {"name": "FATIGUE (Active/Fatigue)", "preds": [], "targets": [], "target_names": ["Active", "Fatigue"]}
    }

    print("Running forward evaluation passes across test tensors...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            # Apply sigmoid activation to get confidence probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.numpy()

            # Separate multi-task outputs using our gradient mask mapping
            for task_idx in range(3):
                for i in range(len(labels)):
                    target_val = labels[i, task_idx]
                    # If target is -1, it belongs to another task's dataset, skip it
                    if target_val != -1:
                        pred_class = 1 if probabilities[i, task_idx] >= 0.5 else 0
                        task_metrics[task_idx]["preds"].append(pred_class)
                        task_metrics[task_idx]["targets"].append(int(target_val))

    # --- GENERATE DETAILED STRUCTURAL CLASSIFICATION REPORTS ---
    print("\n" + "="*60)
    print("               NETRA-DRIVE CORE MODEL METRICS                 ")
    print("="*60)
    
    for task_idx, data in task_metrics.items():
        print(f"\n[TASK PARTITION] : {data['name']}")
        if len(data["targets"]) == 0:
            print(" -> No validation elements processed for this specific task slice.")
            continue
            
        y_true = np.array(data["targets"])
        y_pred = np.array(data["preds"])
        
        # Calculate raw percentage accuracy
        accuracy = np.mean(y_true == y_pred) * 100
        print(f" Raw Stratified Accuracy: {accuracy:.2f}%")
        print("-" * 50)
        
        # Print precision, recall, and f1-score values
        print(classification_report(y_true, y_pred, target_names=data["target_names"], zero_division=0))
        print("=" * 60)

if __name__ == "__main__":
    run_evaluation_pipeline()