import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "datasets", "processed")

# --- 2. MULTI-TASK ON-THE-FLY IMAGE DATASET ---
class NetraDriveDataset(Dataset):
    """
    Custom Dataset that reads our stratified folders and handles non-uniform image 
    sizes by dynamically transforms/resizing them to uniform 224x224 tensors.
    """
    def __init__(self, split="train"):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Task definitions and label mappings
        # -1 represents a missing/masked label for an image from a different dataset
        task_configs = {
            "eyes": {"classes": ["close", "open"], "labels": [0, 1]},
            "yawn": {"classes": ["no_yawn", "yawn"], "labels": [0, 1]},
            "fatigue": {"classes": ["active", "fatigue"], "labels": [0, 1]}
        }

        for task_idx, (task_name, config) in enumerate(task_configs.items()):
            task_path = os.path.join(PROCESSED_DIR, task_name, split)
            if not os.path.exists(task_path):
                continue
                
            for class_name, label_val in zip(config["classes"], config["labels"]):
                class_path = os.path.join(task_path, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Build a dynamic multi-task label mask
                        full_label = [-1, -1, -1] 
                        full_label[list(task_configs.keys()).index(task_name)] = label_val
                        
                        self.samples.append({
                            "path": os.path.join(class_path, img_name),
                            "labels": torch.tensor(full_label, dtype=torch.float32)
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample["path"]).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, sample["labels"]
        except Exception:
            # Fallback for any corrupted images in the dataset
            return torch.zeros((3, 224, 224)), sample["labels"]

# --- 3. MULTI-HEAD NEURAL NETWORK ARCHITECTURE ---
class MultiTaskNetraNet(nn.Module):
    def __init__(self):
        super(MultiTaskNetraNet, self).__init__()
        # Load ultra-lightweight MobileNetV3 for lightning fast CPU/Laptop inference
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.backbone.classifier[0].in_features
        
        # Strip off old classification head
        self.backbone.classifier = nn.Identity()

        # Build independent target predictors
        self.eye_head = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))
        self.yawn_head = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fatigue_head = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        features = self.backbone(x)
        return torch.cat([self.eye_head(features), self.yawn_head(features), self.fatigue_head(features)], dim=1)

# --- 4. PIPELINE TRAINING EXECUTOR ---
def run_training_pipeline():
    print(f"Loading data tensors onto device execution environment: [{DEVICE}]")
    train_loader = DataLoader(NetraDriveDataset("train"), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(NetraDriveDataset("val"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTaskNetraNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction='none') # Permits custom cross-entropy masking
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting model optimization for {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Mask out missing dataset targets dynamically so they don't break backpropagation
            mask = (labels != -1)
            loss = (criterion(outputs, labels) * mask).sum() / (mask.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Completed. Structural Loss Matrix Value: {epoch_loss:.4f}")

    # Export finalized tracking weights
    output_model_path = os.path.join(BASE_DIR, "driver_model.pth")
    torch.save(model.state_dict(), output_model_path)
    print(f"\nOptimization complete! Weights exported safely to: {output_model_path}")

if __name__ == "__main__":
    run_training_pipeline()