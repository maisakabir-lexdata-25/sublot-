import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_coco_json(json_path, images_src_dir, output_images_dir, output_labels_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Map image IDs to file names and dimensions
    image_map = {img['id']: img for img in data['images']}
    
    # Map category IDs to YOLO class IDs (assuming only one class 'sublot')
    # Roboflow export categories: [{'id': 0, 'name': 'sublots', 'supercategory': 'none'}, {'id': 1, 'name': 'sublot', 'supercategory': 'sublots'}]
    # We'll map everything to class 0
    cat_map = {cat['id']: 0 for cat in data['categories']}

    # Group annotations by image ID
    annotations_map = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_map:
            annotations_map[img_id] = []
        annotations_map[img_id].append(ann)

    print(f"Processing {len(image_map)} images from {json_path}...")
    for img_id, img_info in tqdm(image_map.items()):
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Copy image
        src_image_path = Path(images_src_dir) / file_name
        dest_image_path = Path(output_images_dir) / file_name
        if src_image_path.exists():
            shutil.copy(src_image_path, dest_image_path)
        else:
            print(f"Warning: Image {src_image_path} not found.")
            continue

        # Create YOLO label file
        label_name = Path(file_name).stem + ".txt"
        label_path = Path(output_labels_dir) / label_name
        
        with open(label_path, 'w') as f:
            if img_id in annotations_map:
                for ann in annotations_map[img_id]:
                    # segmentation is list of lists of coordinates
                    if 'segmentation' in ann and ann['segmentation']:
                        # Use first category or mapped category
                        yolo_cls = cat_map.get(ann['category_id'], 0)
                        
                        for seg in ann['segmentation']:
                            # Normalize coordinates
                            normalized = []
                            for i in range(0, len(seg), 2):
                                x = seg[i] / width
                                y = seg[i+1] / height
                                normalized.append(f"{x:.6f} {y:.6f}")
                            
                            f.write(f"{yolo_cls} {' '.join(normalized)}\n")

def main():
    root = Path("c:/Users/user/Downloads/sublot/sublot-")
    coco_root = root / "sublot.v1i.coco"
    output_root = root / "dataset"
    
    # Process Train
    convert_coco_json(
        coco_root / "train" / "_annotations.coco.json",
        coco_root / "train",
        output_root / "train" / "images",
        output_root / "train" / "labels"
    )
    
    # Process Valid
    convert_coco_json(
        coco_root / "valid" / "_annotations.coco.json",
        coco_root / "valid",
        output_root / "val" / "images",
        output_root / "val" / "labels"
    )

    # Create data.yaml
    data_yaml = {
        'path': str(output_root),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['sublot']
    }
    
    import yaml
    with open(root / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Successfully converted dataset and created {root / 'data.yaml'}")

if __name__ == "__main__":
    main()
