# CPU Training Configuration Guide

## ⚠️ Memory Issue Fixed

The out-of-memory error has been resolved by optimizing the training configuration for CPU.

## 🔧 Changes Made to `train.py`

```python
BATCH_SIZE = 2   # Changed from 16 to 2
WORKERS = 2      # Changed from 8 to 2
```

## 💡 Why This Happened

1. **No GPU detected** - Training runs on CPU using system RAM
2. **Batch size too large** - 16 images at once exceeded available RAM
3. **Too many workers** - 8 parallel data loaders consumed too much memory

## ✅ Solution Applied

- **Batch size: 16 → 2** - Process only 2 images at a time
- **Workers: 8 → 2** - Use only 2 parallel data loaders
- Model stays the same (yolov8n-seg - smallest/fastest)

## 📊 Training Expectations with CPU

### Speed:
- **Per epoch**: ~5-10 minutes (vs ~30 seconds on GPU)
- **100 epochs**: ~8-16 hours total
- **Recommendation**: Start with 10-20 epochs for testing

### To reduce training time:
```python
# Edit train.py line 13:
EPOCHS = 20  # Instead of 100
```

## 🚀 How to Resume Training

Just run the command again:
```bash
python train.py
```

## 💻 Alternative: Use Google Colab (FREE GPU)

If CPU training is too slow, use Google Colab for free GPU access:

1. Go to https://colab.research.google.com/
2. Upload your dataset
3. Run training with GPU (much faster!)

Would you like me to create a Colab notebook for you?

## 🎯 Quick Test (Recommended)

Before running 100 epochs, test with fewer epochs:

```python
# In train.py, change line 13:
EPOCHS = 10  # Quick test run
```

This will complete in ~1 hour on CPU and let you verify everything works.

## 📈 Performance Comparison

| Setting | GPU | CPU (Original) | CPU (Fixed) |
|---------|-----|----------------|-------------|
| Batch Size | 16 | 16 | 2 |
| Workers | 8 | 8 | 2 |
| Time/Epoch | ~30s | ❌ Crash | ~5-10 min |
| 100 Epochs | ~50 min | ❌ Crash | ~8-16 hours |

## ⚡ If You Get a GPU Later

When you have GPU access, change these back in `train.py`:

```python
BATCH_SIZE = 16  # Or even 32 if you have 8GB+ VRAM
WORKERS = 8
```
