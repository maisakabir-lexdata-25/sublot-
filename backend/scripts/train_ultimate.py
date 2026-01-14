
import os
import time
import torch
import torchvision
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ultralytics import YOLO

# Configuration
EPOCHS = 30
BATCH_SIZE = 2 # Lowered for dual-model overhead
IMAGE_SIZE = 640
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2 

print(f"🚀 Initializing ULTIMATE HYBRID Training (YOLOv11 + Mask R-CNN + SAM Logic)")

# -------------------------------------------------------------------
# Ultimate Hybrid Trainer
# -------------------------------------------------------------------

class SublotDataset(Dataset):
    def __init__(self, root, split='train', img_size=640):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.imgs_dir = self.root / split / 'images'
        self.labels_dir = self.root / split / 'labels'
        self.img_files = sorted(list(self.imgs_dir.glob('*.jpg')) + list(self.imgs_dir.glob('*.png')))

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        label_path = self.labels_dir / (img_path.stem + '.txt')
        boxes, masks, labels = [], [], []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                obj_label = class_id + 1 
                coords = parts[1:]
                poly_points = np.array(coords).reshape(-1, 2)
                poly_points[:, 0] *= self.img_size
                poly_points[:, 1] *= self.img_size
                poly_points = poly_points.astype(np.int32)
                
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_points], 1)
                
                x_min, y_min = np.min(poly_points, axis=0)
                x_max, y_max = np.max(poly_points, axis=0)
                
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    masks.append(mask)
                    labels.append(obj_label)

        img_tensor = torchvision.transforms.functional.to_tensor(img_resized)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8) if masks else torch.zeros((0, self.img_size, self.img_size)),
        }
        return img_tensor, target

    def __len__(self):
        return len(self.img_files)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # 1. Prepare YOLOv11 for the first stage (Transfer learning)
    print("Pre-loading YOLOv11m-seg for primary features...")
    yolo_model = YOLO("yolo11m-seg.pt")
    
    # 2. Build Mask R-CNN for the second stage (Precision)
    print("Building Mask R-CNN precision head...")
    mask_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = mask_model.roi_heads.box_predictor.cls_score.in_features
    mask_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = mask_model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    mask_model.to(DEVICE)

    # 3. Setup Dataset
    data_path = "e:/sublot-/dataset"
    dataset = SublotDataset(data_path, split='train', img_size=IMAGE_SIZE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 4. Hybrid Optimization logic
    # We first finetune the YOLO model (Ultralytics handles its own training loop well)
    print("\n[Phase 1] Finetuning YOLOv11 primary segmenter...")
    yolo_model.train(
        data="data.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=0,
        project="ultimate_hybrid",
        name="yolo_stage",
        exist_ok=True,
        verbose=False
    )

    # 5. [Phase 2] Train Mask R-CNN precision head
    print("\n[Phase 2] Training Mask R-CNN precision refiner...")
    optimizer = torch.optim.AdamW(mask_model.parameters(), lr=0.00005, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, total_steps=len(data_loader) * EPOCHS)

    for epoch in range(EPOCHS):
        mask_model.train()
        epoch_loss = 0
        pbar = enumerate(data_loader)
        for i, (images, targets) in pbar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = mask_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += losses.item()
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {i}/{len(data_loader)} | Loss: {losses.item():.4f}")

        print(f"✅ Epoch {epoch+1} Completed. Avg Loss: {epoch_loss/len(data_loader):.4f}")

    # 6. Save Ultimate Weights
    save_path = "ultimate_hybrid_refiner.pth"
    torch.save(mask_model.state_dict(), save_path)
    print(f"\n🏆 Ultimate Hybrid Training Finished!")
    print(f"YOLO Stage Weights: ultimate_hybrid/yolo_stage/weights/best.pt")
    print(f"Mask Refiner Weights: {save_path}")

if __name__ == "__main__":
    main()
