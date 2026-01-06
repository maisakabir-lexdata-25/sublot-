"""
YOLOv8 Segmentation Prediction Script
Run inference on images and visualize polygon segmentation results
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# Default configuration
DEFAULT_MODEL = "sublot_segmentation/yolov8_seg/weights/best.pt"
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
    
    return parser.parse_args()

def draw_segmentation(image, result):
    """Draw segmentation masks and polygons on image"""
    annotated = image.copy()
    
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()
        
        # Generate colors for each instance
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
        
        for idx, (mask, box) in enumerate(zip(masks, boxes)):
            # Get confidence and class
            conf = box[4]
            cls = int(box[5])
            
            # Resize mask to image size
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # Create colored overlay
            color = colors[idx].tolist()
            overlay = annotated.copy()
            overlay[mask_bool] = color
            
            # Blend with original image
            alpha = 0.4
            annotated = cv2.addWeighted(annotated, 1, overlay, alpha, 0)
            
            # Draw polygon contour
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, color, 2)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"sublot {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return annotated

def main():
    args = parse_args()
    
    print("=" * 60)
    print("YOLOv8 Segmentation Prediction")
    print("=" * 60)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Error: Model not found at {model_path}")
        print(f"\nPlease train the model first or specify correct path:")
        print(f"  python predict.py --model path/to/weights.pt")
        return
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # STRICT YOLO-SEG ENFORCEMENT
    if model.task != 'segment':
        print(f"ERROR: Model '{args.model}' is a '{model.task}' model.")
        print("This script is strictly for YOLO-SEG (segmentation) models.")
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
        
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.img_size,
            verbose=False,
            save=False,
        )
        
        # Draw segmentation
        annotated = draw_segmentation(image, results[0])
        
        # Save result
        output_file = output_path / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(output_file), annotated)
        
        # Save txt if requested
        if args.save_txt and results[0].masks is not None:
            txt_file = output_path / f"{img_path.stem}.txt"
            with open(txt_file, 'w') as f:
                masks = results[0].masks.xy
                boxes = results[0].boxes.data.cpu().numpy()
                for mask, box in zip(masks, boxes):
                    conf = box[4]
                    cls = int(box[5])
                    # Normalize coordinates
                    h, w = image.shape[:2]
                    mask_norm = mask / np.array([w, h])
                    # Write to file
                    line = f"{cls}"
                    for point in mask_norm:
                        line += f" {point[0]:.6f} {point[1]:.6f}"
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
