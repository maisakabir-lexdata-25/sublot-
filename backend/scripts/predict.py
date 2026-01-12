"""
YOLOv11 Segmentation Prediction Script
Run inference on images and visualize polygon segmentation results
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.utils.post_process import (
    detect_roads, subtract_roads, get_tiles_metadata, 
    is_centroid_in_center, split_merged_mask_watershed, 
    is_rectangular, non_max_suppression_shapely, is_on_border
)

# Default configuration
DEFAULT_MODEL = "backend/models/yolov11_sublot/yolov11m_final/weights/best.pt"
DEFAULT_SOURCE = "test/images"
DEFAULT_OUTPUT = "predictions"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMG_SIZE = 640

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Prediction")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help="Path to image or directory of images")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output directory for predictions")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE,
                        help="Inference image size")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results to txt files")
    parser.add_argument("--save-conf", action="store_true",
                        help="Save confidence scores in txt files")
    parser.add_argument("--show", action="store_true",
                        help="Display results")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to run on (e.g. 0 or cpu)")
    parser.add_argument("--tile", action="store_true",
                        help="Use tiled inference with overlap")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Size of each tile")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Overlap between tiles (0.25-0.30)")
    
    return parser.parse_args()

def draw_results(image, processed_polys):
    """Draw processed polygons on image"""
    annotated = image.copy()
    
    for poly, conf, cls in processed_polys:
        color = [int(c) for c in np.random.randint(0, 255, size=3)]
        
        # Draw the region as an overlay
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [poly.astype(np.int32)], color)
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
        
        # Draw boundary
        cv2.polylines(annotated, [poly.astype(np.int32)], True, color, 2)
        
        # Draw label at center of polygon
        M = cv2.moments(poly)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            label = f"Sublot: {conf:.2f}"
            cv2.putText(annotated, label, (cX - 40, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated

# Helper functions removed - now using backend.utils.post_process

def main():
    args = parse_args()
    
    print("=" * 60)
    print("YOLOv11 Segmentation Prediction")
    print("=" * 60)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Error: Model not found at {model_path}")
        return
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    if model.task != 'segment':
        print(f"ERROR: Model '{args.model}' is a '{model.task}' model.")
        return
    
    # Check source
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"✗ Error: Source not found at {source_path}")
        return
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print("=" * 60)
    
    # Get image files
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
    
    if len(image_files) == 0:
        print("✗ No images found!")
        return
    
    print(f"\nProcessing {len(image_files)} images...\n")
    
    # Run predictions
    for img_path in tqdm(image_files, desc="Predicting"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        
        processed_polys = [] # List of (polygon, conf, cls)
        
        if args.tile:
            tiles = get_tiles_metadata(w, h, args.tile_size, args.overlap)
            for tx, ty, tw, th in tiles:
                tile_img = image[ty:ty+th, tx:tx+tw]
                
                # YOLO segmentation inference on tile
                results = model.predict(
                    source=tile_img,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.img_size,
                    verbose=False,
                    device=args.device,
                )[0]
                
                if results.masks is not None:
                    masks = results.masks.data.cpu().numpy()
                    boxes = results.boxes.data.cpu().numpy()
                    xy_polys = results.masks.xy
                    
                    for mask, box, poly in zip(masks, boxes, xy_polys):
                        # Construct global polygon
                        global_poly = poly + np.array([tx, ty])
                        
                        # Apply Tile Center Filtering
                        if not is_centroid_in_center(global_poly, (tx, ty, tw, th), args.overlap, (w, h)):
                            continue
                        
                        # Road Detection and Subtraction (Per-Tile)
                        mask_resized = cv2.resize(mask, (tw, th))
                        mask_bool = mask_resized > 0.5
                        
                        # Rule: Drop if touching TILE boundary (ensures completeness and no crossing)
                        if is_on_border(mask_bool):
                            continue
                        
                        tile_road_mask = detect_roads(tile_img)
                        mask_uint8 = (mask_bool * 255).astype(np.uint8)
                        field_masks = subtract_roads(mask_uint8, tile_road_mask)
                        
                        conf = box[4]
                        cls = int(box[5])
                        
                        for fm in field_masks:
                            # Rule: Re-check border after road subtraction (splitting might create edge-touching shards)
                            if is_on_border(fm):
                                continue
                                
                            # Split merged sublots (on the tile-sized mask)
                            sub_polys_tile = split_merged_mask_watershed(fm)
                            
                            for sp in sub_polys_tile:
                                # Rule: Topological / Quality check
                                area = cv2.contourArea(sp)
                                if area < 200: # Slightly stricter area check
                                    continue
                                
                                global_sp = sp + np.array([tx, ty])
                                processed_polys.append((global_sp, conf, cls))
        else:
            # Normal inference
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.img_size,
                verbose=False,
                device=args.device,
            )[0]
            
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                boxes = results.boxes.data.cpu().numpy()
                
                for mask, box in zip(masks, boxes):
                    # Rule: Complete inside the image
                    if is_on_border(mask):
                        continue
                    
                    # Road Detection and Subtraction
                    road_mask = detect_roads(image)
                    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
                    field_masks = subtract_roads(mask_uint8, road_mask)
                    
                    conf = box[4]
                    cls = int(box[5])
                    for fm in field_masks:
                        # Rule: Topological / Quality check
                        if fm.sum() < 200:
                            continue
                        if is_on_border(fm):
                            continue
                            
                        sub_polys = split_merged_mask_watershed(fm)
                        for sp in sub_polys:
                            if cv2.contourArea(sp) < 200:
                                continue
                            processed_polys.append((sp, conf, cls))
        
        # Draw results
        annotated = draw_results(image, processed_polys)
        
        # Save result
        output_file = output_path / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(output_file), annotated)
        
        # Save txt if requested
        if args.save_txt:
            txt_file = output_path / f"{img_path.stem}.txt"
            with open(txt_file, 'w') as f:
                for poly, conf, cls in processed_polys:
                    # Normalize coordinates
                    normalized = []
                    for pt in poly:
                        normalized.append(f"{pt[0]/w:.6f} {pt[1]/h:.6f}")
                    
                    line = f"{cls} {' '.join(normalized)}"
                    if args.save_conf:
                        line += f" {conf:.6f}"
                    f.write(line + "\n")
        
        # Show if requested
        if args.show:
            cv2.imshow("Prediction", annotated)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    
    if args.show:
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✓ Prediction completed!")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print(f"Total images processed: {len(image_files)}")

if __name__ == "__main__":
    main()
