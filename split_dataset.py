import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    base_path = Path(base_dir)
    output_path = Path(output_dir)

    # Ensure ratios sum to 1.0 (with a small tolerance for floating point)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Gather all image paths and their corresponding labels (class names)
    image_paths = []
    labels = []
    
    # Get all class directories
    classes = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    
    for class_name in classes:
        class_dir = base_path / class_name
        for img_path in class_dir.glob('*'):
            if img_path.is_file():
                image_paths.append(img_path)
                labels.append(class_name)

    if not image_paths:
        print(f"No images found in {base_path}")
        return

    # First split: Separate out the test set (10% of total data)
    # The remaining data will be for train and validation (90% of total data)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_ratio, stratify=labels, random_state=42
    )

    # Second split: Separate the remaining data into train (80% of total) and val (10% of total)
    # To get 10% of total from the 90% remaining, the relative test_size is 10/90 (or val_ratio / (train_ratio + val_ratio))
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_relative_ratio, stratify=train_val_labels, random_state=42
    )

    # Function to copy files to their respective dataset splits
    def copy_files(paths, split_name):
        print(f"Copying {len(paths)} files to {split_name}...")
        split_dir = output_path / split_name
        
        # Create directories for each class
        for class_name in classes:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
            
        for path in paths:
            class_name = path.parent.name
            dest_path = split_dir / class_name / path.name
            shutil.copy2(path, dest_path)

    # Execute the copying process
    copy_files(train_paths, 'train')
    copy_files(val_paths, 'val')
    copy_files(test_paths, 'test')

    # Display counts
    print("\nDataset Split Summary:")
    print("-" * 75)
    print(f"{'Class Name':<25} | {'Train (80%)':<12} | {'Val (10%)':<12} | {'Test (10%)':<12}")
    print("-" * 75)
    
    total_train, total_val, total_test = 0, 0, 0
    for class_name in classes:
        train_count = len(list((output_path / 'train' / class_name).glob('*')))
        val_count = len(list((output_path / 'val' / class_name).glob('*')))
        test_count = len(list((output_path / 'test' / class_name).glob('*')))
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        print(f"{class_name:<25} | {train_count:<12} | {val_count:<12} | {test_count:<12}")
        
    print("-" * 75)
    print(f"{'TOTAL':<25} | {total_train:<12} | {total_val:<12} | {total_test:<12}")
    print("-" * 75)
    print("\nDataset successfully split and copied!")

if __name__ == "__main__":
    # Define directories relative to where script is executed
    base_directory = "train_images"
    output_directory = "dataset"
    
    # Check if train_images exists before running
    if not Path(base_directory).exists():
        print(f"Error: The directory '{base_directory}' was not found.")
        print("Please ensure you run this script from the PADDY-DISEASE-CLASSIFICATION folder.")
    else:
        # Run the split
        split_dataset(base_directory, output_directory)
