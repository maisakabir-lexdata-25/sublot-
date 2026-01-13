
import os
import glob
import time
import numpy as np
import cv2
import torch
import yaml
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

# Configuration
EPOCHS = 20
BATCH_SIZE = 4
IMAGE_SIZE = 640
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2  # 0=background, 1=sublot (assumed based on single class in data.yaml)

print(f"Using device: {DEVICE}")

def get_mask_rcnn_model(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

class SublotDataset(Dataset):
    def __init__(self, root, split='train', img_size=640):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        
        # Parse data.yaml to verify paths (optional but good practice)
        # Assuming standard structure based on checking: dataset/train/images etc.
        self.imgs_dir = self.root / split / 'images'
        self.labels_dir = self.root / split / 'labels'
        
        self.img_files = sorted(list(self.imgs_dir.glob('*.jpg')) + list(self.imgs_dir.glob('*.png')))
        print(f"Found {len(self.img_files)} images in {self.imgs_dir}")

    def __getitem__(self, idx):
        # Load Image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original H, W
        h_orig, w_orig = img.shape[:2]
        
        # Resize Image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Load Labels
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
        boxes = []
        masks = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                
                # We assume class_id 0 is the sublot, mapping to model class 1
                # Background is class 0 in the model.
                obj_label = class_id + 1 
                
                coords = parts[1:]
                # Reshape to list of (x, y)
                poly_points = np.array(coords).reshape(-1, 2)
                
                # Denormalize coordinates to ORIGINAL image size first
                poly_points[:, 0] *= w_orig
                poly_points[:, 1] *= h_orig
                
                # Scale coordinates to RESIZED image size
                scale_x = self.img_size / w_orig
                scale_y = self.img_size / h_orig
                
                poly_points[:, 0] *= scale_x
                poly_points[:, 1] *= scale_y
                
                poly_points = poly_points.astype(np.int32)
                
                # Create Mask
                # Mask needs to be uint8
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_points], 1)
                
                # Create BBox
                x_min = np.min(poly_points[:, 0])
                y_min = np.min(poly_points[:, 1])
                x_max = np.max(poly_points[:, 0])
                y_max = np.max(poly_points[:, 1])
                
                # Handle edge case where bbox is degenerate
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                boxes.append([x_min, y_min, x_max, y_max])
                masks.append(mask)
                labels.append(obj_label)

        # Convert to Tensors
        # Normalize image: 0-255 -> 0-1
        img_tensor = F.to_tensor(img_resized)
        
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            masks_tensor = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            
            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "masks": masks_tensor,
                "image_id": torch.tensor([idx]),
                "area": area,
                "iscrowd": iscrowd
            }
        else:
            # Negative sample (no objects)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        return img_tensor, target

    def __len__(self):
        return len(self.img_files)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Setup dataset
    data_path = "e:/sublot-/dataset"
    if not os.path.exists(data_path):
        print(f"Error: Dataset path {data_path} not found.")
        return

    dataset = SublotDataset(data_path, split='train', img_size=IMAGE_SIZE)
    # subset for validation if needed, or separate val loader
    dataset_val = SublotDataset(data_path, split='val', img_size=IMAGE_SIZE)

    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Setup Model
    model = get_mask_rcnn_model(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        num_batches = len(data_loader)
        
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            
            epoch_loss += loss_value

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: [{epoch+1}/{EPOCHS}], Batch: [{i}/{num_batches}], Loss: {loss_value:.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} Completed in {time.time() - start_time:.1f}s, Avg Loss: {avg_loss:.4f}")
        
    # Save Model
    save_path = Path("mask_rcnn_sublot_20e.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Finished. Model saved to {save_path.absolute()}")

if __name__ == "__main__":
    main()
