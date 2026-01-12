import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datasets import Dataset, Features, Image as DatasetsImage
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation,
    TrainingArguments, 
    Trainer
)
import evaluate

# --- Configuration ---
MODEL_ID = "nvidia/mit-b0"  # Smallest SegFormer model
WORKING_DIR = Path("segformer_sublot_v2")
DATASET_DIR = Path(".")
BATCH_SIZE = 2
LR = 6e-5
EPOCHS = 35 # Sufficient for good convergence

# --- Dataset Loading ---
def load_data(split):
    split_dir = DATASET_DIR / split
    mask_dir = split_dir / "masks"
    
    if not mask_dir.exists():
        return None
        
    data = {"image": [], "label": []}
    
    # Handle both flattened and nested structures
    img_dir = split_dir / "images"
    if not img_dir.exists():
        img_dir = split_dir
        
    mask_files = list(mask_dir.glob("*.png"))
    for mask_path in mask_files:
        img_path = img_dir / f"{mask_path.stem}.jpg" # Common format
        if not img_path.exists():
             img_path = img_dir / f"{mask_path.stem}.png"
             
        if img_path.exists():
            data["image"].append(str(img_path))
            data["label"].append(str(mask_path))
            
    return Dataset.from_dict(data).cast_column("image", DatasetsImage()).cast_column("label", DatasetsImage())

print("Loading dataset...")
train_dataset = load_data("train")
test_dataset = load_data("test")

if train_dataset is None:
    print("Error: Train dataset not found. Run dataset_to_masks.py first.")
    exit()

# --- Processor & Transforms ---
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
processor.do_reduce_labels = False # We have 0: background, 1: sublot

def train_transforms(example_batch):
    images = [x.convert("RGB") for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = processor(images, labels, return_tensors="pt", size={"height": 384, "width": 384})
    return inputs

train_dataset.set_transform(train_transforms)
if test_dataset:
    test_dataset.set_transform(train_transforms)

# --- Model & Metrics ---
id2label = {0: "background", 1: "sublot"}
label2id = {v: k for k, v in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # Scale the logits to the size of the label
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=False,
        )
        
        # Format metrics
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics

# --- Training ---
training_args = TrainingArguments(
    output_dir=str(WORKING_DIR),
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, # Higher for stability on limited CPU RAM
    save_total_limit=2,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=500,
    logging_steps=10,
    remove_unused_columns=False,
    push_to_hub=False,
    use_cpu=True if not torch.cuda.is_available() else False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model(str(WORKING_DIR / "final_model"))
processor.save_pretrained(str(WORKING_DIR / "final_model"))
print(f"Model saved to {WORKING_DIR / 'final_model'}")
