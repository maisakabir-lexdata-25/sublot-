"""
Dataset Split Script for YOLOv11 Segmentation
Splits images and labels into train/val/test sets
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_ROOT = Path(__file__).parent
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Output directories
TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"
VAL_IMAGES = DATASET_ROOT / "val" / "images"
VAL_LABELS = DATASET_ROOT / "val" / "labels"
TEST_IMAGES = DATASET_ROOT / "test" / "images"
TEST_LABELS = DATASET_ROOT / "test" / "labels"

def create_directories():
    """Create train/val/test directory structure"""
    for dir_path in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, TEST_IMAGES, TEST_LABELS]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✓ Created directory structure")

def get_image_files():
    """Get all image files from the images directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in IMAGES_DIR.iterdir() 
                   if f.suffix.lower() in image_extensions]
    return image_files

def split_dataset(image_files, seed=42):
    """Split dataset into train/val/test sets"""
    random.seed(seed)
    random.shuffle(image_files)
    
    total = len(image_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    return train_files, val_files, test_files

def copy_files(image_files, dest_images, dest_labels, split_name):
    """Copy image and label files to destination directories"""
    print(f"\nCopying {split_name} files...")
    
    copied_images = 0
    copied_labels = 0
    missing_labels = []
    
    for img_file in tqdm(image_files, desc=split_name):
        # Copy image
        shutil.copy2(img_file, dest_images / img_file.name)
        copied_images += 1
        
        # Copy corresponding label
        label_file = LABELS_DIR / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, dest_labels / label_file.name)
            copied_labels += 1
        else:
            missing_labels.append(img_file.name)
    
    print(f"  ✓ Copied {copied_images} images")
    print(f"  ✓ Copied {copied_labels} labels")
    
    if missing_labels:
        print(f"  ⚠ Warning: {len(missing_labels)} images without labels:")
        for name in missing_labels[:5]:
            print(f"    - {name}")
        if len(missing_labels) > 5:
            print(f"    ... and {len(missing_labels) - 5} more")

def main():
    print("=" * 60)
    print("YOLOv11 Dataset Split Script")
    print("=" * 60)
    
    # Check if source directories exist
    if not IMAGES_DIR.exists():
        print(f"✗ Error: Images directory not found: {IMAGES_DIR}")
        return
    
    if not LABELS_DIR.exists():
        print(f"✗ Error: Labels directory not found: {LABELS_DIR}")
        return
    
    # Get all image files
    image_files = get_image_files()
    print(f"\nFound {len(image_files)} images")
    
    if len(image_files) == 0:
        print("✗ Error: No images found!")
        return
    
    # Create directory structure
    create_directories()
    
    # Split dataset
    train_files, val_files, test_files = split_dataset(image_files)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} images ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"  Val:   {len(val_files)} images ({len(val_files)/len(image_files)*100:.1f}%)")
    print(f"  Test:  {len(test_files)} images ({len(test_files)/len(image_files)*100:.1f}%)")
    
    # Copy files
    copy_files(train_files, TRAIN_IMAGES, TRAIN_LABELS, "Train")
    copy_files(val_files, VAL_IMAGES, VAL_LABELS, "Val")
    copy_files(test_files, TEST_IMAGES, TEST_LABELS, "Test")
    
    print("\n" + "=" * 60)
    print("✓ Dataset split completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the data.yaml file")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start training: python train.py")

if __name__ == "__main__":
    main()
