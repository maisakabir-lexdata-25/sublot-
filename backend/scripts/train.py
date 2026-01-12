"""
YOLOv11-SEG Sublot Segmentation Training Script
Train a YOLOv11-seg model for sublot detection with polygon segmentation
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

# Configuration for RTX 4060
MODEL_SIZE = "m"  # Medium model for high accuracy
EPOCHS = 100
BATCH_SIZE = 8  # Optimized for RTX 4060 8GB
IMAGE_SIZE = 640
DEVICE = 0  # Use GPU 0
PROJECT_NAME = "yolov11_sublot"
EXPERIMENT_NAME = "yolov11m_final"

# Advanced settings
PATIENCE = 50  # Early stopping patience
SAVE_PERIOD = -1  # Only save last and best (avoid frequent I/O)
WORKERS = 8  # Higher workers for faster data loading on modern CPUs
OPTIMIZER = "auto"  # Options: SGD, Adam, AdamW, auto
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# Data augmentation
AUGMENT = True
HSV_H = 0.015  # HSV-Hue augmentation
HSV_S = 0.7    # HSV-Saturation augmentation
HSV_V = 0.4    # HSV-Value augmentation
DEGREES = 15.0  # Rotation (+/- deg) - Helps with varied field orientations
TRANSLATE = 0.1  # Translation (+/- fraction)
SCALE = 0.5    # Scale (+/- gain)
SHEAR = 0.0    # Shear (+/- deg)
PERSPECTIVE = 0.0001  # Perspective (+/- fraction) - Helps with slanted views
FLIPUD = 0.0   # Flip up-down probability
FLIPLR = 0.5   # Flip left-right probability
MOSAIC = 1.0   # Mosaic augmentation probability
MIXUP = 0.0    # MixUp augmentation probability

def check_dataset():
    """Verify dataset structure and configuration"""
    data_yaml = Path("data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError("data.yaml not found! Please create it first.")
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print("=" * 60)
    print("Dataset Configuration:")
    print("=" * 60)
    print(f"Path: {data.get('path', 'Not specified')}")
    print(f"Train: {data.get('train', 'Not specified')}")
    print(f"Val: {data.get('val', 'Not specified')}")
    print(f"Classes: {data.get('nc', 'Not specified')}")
    print(f"Names: {data.get('names', 'Not specified')}")
    print("=" * 60)
    
    return data_yaml

def main():
    print("\n" + "=" * 60)
    print("YOLOv11 Segmentation Training")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU detected, training will use CPU (slower)")
    
    # Verify dataset
    data_yaml = check_dataset()
    
    # Initialize model (STRICT YOLO-SEG)
    model_name = f"yolo11{MODEL_SIZE}-seg.pt"
    print(f"\nInitializing YOLO11-SEG model: {model_name}")
    model = YOLO(model_name)
    
    # Verify task is segment
    if model.task != 'segment':
        raise ValueError(f"Error: Model task is '{model.task}', but YOLO-SEG is required.")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Augmentation: {AUGMENT}")
    print("=" * 60)
    
    # Train the model
    print("\nStarting training...\n")
    
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        cache=False,  # Force reload dataset (ignore stale cache)
        
        # Optimization
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        
        # Training settings
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        workers=WORKERS,
        pretrained=True,
        verbose=True,
        
        # Augmentation
        augment=AUGMENT,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        
        # Validation
        val=True,
        plots=True,
        save=True,
        save_json=False,  # Disable to avoid I/O issues
        exist_ok=True,    # Allow overwriting existing project folder
    )
    
    print("\n" + "=" * 60)
    print("✓ Training completed!")
    print("=" * 60)
    
    # Print results location
    save_dir = Path(PROJECT_NAME) / EXPERIMENT_NAME
    print(f"\nResults saved to: {save_dir}")
    print(f"  - Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"  - Last weights: {save_dir / 'weights' / 'last.pt'}")
    print(f"  - Metrics: {save_dir / 'results.csv'}")
    print(f"  - Plots: {save_dir}")
    
    print("\nNext steps:")
    print("1. Review training metrics in results.csv")
    print("2. Check plots for loss curves and metrics")
    print("3. Run validation: python validate.py")
    print("4. Test predictions: python predict.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
