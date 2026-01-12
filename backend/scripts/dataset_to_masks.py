import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_yolo_to_mask(img_path, label_path, output_path):
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    
    # Create empty mask (0: background)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Read label file
    if not label_path.exists():
        # Save empty mask if no labels
        cv2.imwrite(str(output_path), mask)
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        # YOLO format: class x1 y1 x2 y2 ...
        # class = parts[0]
        points = [float(x) for x in parts[1:]]
        
        # Reshape to (n, 2) and scale to image size
        points = np.array(points).reshape(-1, 2)
        points[:, 0] *= w
        points[:, 1] *= h
        points = points.astype(np.int32)
        
        # Fill polygon in mask (1: sublot)
        cv2.fillPoly(mask, [points], color=1)
        
        # Draw a black boundary (0) around each sublot to keep them separated
        # Using thickness 3 for a clear 6-pixel gap between adjacent plots
        cv2.polylines(mask, [points], True, color=0, thickness=3)
        
    # Save mask
    cv2.imwrite(str(output_path), mask)

def main():
    # Setup paths
    root = Path(".")
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"Converting {split} split...")
        split_dir = root / split
        mask_dir = split_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        if not split_dir.exists():
            print(f"Skipping {split} - directory not found")
            continue
            
        # Handle both structures: split/images/ or just split/
        img_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        
        if not img_dir.exists():
            img_dir = split_dir
        if not label_dir.exists():
            label_dir = split_dir
            
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in img_dir.glob("*") if f.suffix.lower() in image_extensions]
        
        for img_path in tqdm(images):
            # Check for label file
            label_path = label_dir / f"{img_path.stem}.txt"
            output_path = mask_dir / f"{img_path.stem}.png"
            
            convert_yolo_to_mask(img_path, label_path, output_path)

if __name__ == "__main__":
    main()
