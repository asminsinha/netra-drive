import os
import shutil
import random
from tqdm import tqdm

def setup_stratified_dataset():
    # Define primary root working spaces
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "datasets", "raw")
    processed_dir = os.path.join(base_dir, "datasets", "processed")

    # Map target source structures
    dataset_2_path = os.path.join(raw_dir, "dataset_2_features")
    dataset_3_path = os.path.join(raw_dir, "dataset_3_holistic")

    # Check for presence of source data before structural build
    if not os.path.exists(dataset_2_path) or not os.path.exists(dataset_3_path):
        print(f"[ERROR] Source files missing in raw datasets paths.\nVerify download paths matching:\n- {dataset_2_path}\n- {dataset_3_path}")
        return

    # Define tasks and corresponding categorical folders
    tasks = {
        "eyes": {"source": dataset_2_path, "classes": ["open", "close"]},
        "yawn": {"source": dataset_2_path, "classes": ["no_yawn", "yawn"]},
        "fatigue": {"source": dataset_3_path, "classes": ["active", "fatigue"]}
    }

    # Splits ratio configuration: 70% Train, 15% Val, 15% Test
    splits = ["train", "val", "test"]
    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    # Set seed value to preserve deterministic validation across runs
    random.seed(42)

    print("Initializing Multi-Task Dataset Stratification Engine...")

    for task_name, task_meta in tasks.items():
        print(f"\nProcessing Task Partition Category: [{task_name.upper()}]")
        
        for cls in task_meta["classes"]:
            src_class_path = os.path.join(task_meta["source"], cls)
            if not os.path.exists(src_class_path):
                print(f"[WARNING] Skipping missing tracking subdirectory: {src_class_path}")
                continue

            # Gather all images in this folder
            all_files = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(all_files)

            # Compute split indices mathematically
            total_count = len(all_files)
            train_end = int(total_count * split_ratios["train"])
            val_end = train_end + int(total_count * split_ratios["val"])

            file_splits = {
                "train": all_files[:train_end],
                "val": all_files[train_end:val_end],
                "test": all_files[val_end:]
            }

            # Distribute files to target directories
            for phase in splits:
                dest_folder = os.path.join(processed_dir, task_name, phase, cls)
                os.makedirs(dest_folder, exist_ok=True)

                print(f" -> Mapping {len(file_splits[phase])} elements to {task_name}/{phase}/{cls}")
                for filename in tqdm(file_splits[phase], desc=f"{phase}-{cls}", leave=False):
                    src_file = os.path.join(src_class_path, filename)
                    dest_file = os.path.join(dest_folder, filename)
                    
                    # Copy the files over securely
                    shutil.copy2(src_file, dest_file)

    print("\nDataset split complete. Clean 70/15/15 stratification generated inside your 'datasets/processed/' folder.")

if __name__ == "__main__":
    setup_stratified_dataset()