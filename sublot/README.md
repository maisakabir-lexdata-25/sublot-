# YOLOv8 Sublot Segmentation

This project trains a YOLOv8 segmentation model to detect sublots using polygon annotations.

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Split your dataset into train/val/test sets:

```bash
python dataset_split.py
```

This will create the following structure:
```
sublot/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 3. Train the Model

```bash
python train.py
```

**Training Options:**
- Edit `train.py` to change model size (n/s/m/l/x)
- Adjust epochs, batch size, image size
- Configure augmentation parameters

**Model Sizes:**
- `yolov8n-seg`: Fastest, smallest (3.4M params)
- `yolov8s-seg`: Balanced (11.8M params) ⭐ Recommended
- `yolov8m-seg`: More accurate (27.3M params)
- `yolov8l-seg`: Large (46.0M params)
- `yolov8x-seg`: Extra large (71.8M params)

### 4. Validate the Model

```bash
python validate.py
```

Optional arguments:
```bash
python validate.py --split test --save-json
```

### 5. Run Predictions

```bash
python predict.py
```

**Prediction Options:**
```bash
# Predict on specific image
python predict.py --source path/to/image.jpg

# Predict on directory
python predict.py --source test/images

# Custom model and confidence
python predict.py --model path/to/weights.pt --conf 0.5

# Save prediction labels
python predict.py --save-txt --save-conf

# Display results
python predict.py --show
```

## 📁 Project Structure

```
sublot/
├── dataset_split.py      # Split dataset into train/val/test
├── train.py             # Training script
├── validate.py          # Validation script
├── predict.py           # Inference script
├── requirements.txt     # Python dependencies
├── data.yaml           # Dataset configuration
├── classes.txt         # Class names
├── images/             # Original images
├── labels/             # Original annotations (YOLO format)
├── train/              # Training set
├── val/                # Validation set
├── test/               # Test set
└── sublot_segmentation/ # Training outputs
    └── yolov8_seg/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        └── *.png (plots)
```

## 📊 Training Outputs

After training, you'll find:

- **Weights**: `sublot_segmentation/yolov8_seg/weights/`
  - `best.pt`: Best model checkpoint
  - `last.pt`: Last epoch checkpoint

- **Metrics**: `sublot_segmentation/yolov8_seg/results.csv`
  - Training/validation losses
  - mAP scores
  - Precision/recall

- **Plots**: `sublot_segmentation/yolov8_seg/`
  - Loss curves
  - Precision-recall curves
  - Confusion matrix
  - Sample predictions

## 🎯 Annotation Format

Labels are in YOLOv8 segmentation format (normalized coordinates):

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Example:
```
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
```

- `class_id`: Class index (0 for sublot)
- `x1 y1 ... xn yn`: Polygon vertices (normalized 0-1)

## 🔧 Configuration

### Training Hyperparameters

Edit `train.py` to customize:

```python
MODEL_SIZE = "s"        # Model size: n/s/m/l/x
EPOCHS = 100           # Training epochs
BATCH_SIZE = 16        # Batch size
IMAGE_SIZE = 640       # Input image size
LEARNING_RATE = 0.01   # Initial learning rate
PATIENCE = 50          # Early stopping patience
```

### Data Augmentation

```python
AUGMENT = True         # Enable augmentation
FLIPLR = 0.5          # Horizontal flip probability
MOSAIC = 1.0          # Mosaic augmentation
HSV_H = 0.015         # Hue augmentation
HSV_S = 0.7           # Saturation augmentation
HSV_V = 0.4           # Value augmentation
```

## 📈 Performance Metrics

The model is evaluated using:

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

Both box and mask metrics are reported.

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `train.py`
- Use smaller model size (`yolov8n-seg`)
- Reduce `IMAGE_SIZE`

### Low mAP Scores
- Train for more epochs
- Increase dataset size
- Adjust augmentation parameters
- Try larger model size

### Slow Training
- Use GPU instead of CPU
- Increase `BATCH_SIZE` (if memory allows)
- Reduce `IMAGE_SIZE`
- Use smaller model size

## 📚 Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)

## 📝 License

This project uses YOLOv8 from Ultralytics (AGPL-3.0 license).

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!
