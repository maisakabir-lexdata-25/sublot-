"""
YOLOv8 Segmentation Validation Script
Evaluate model performance on validation/test set
"""

from ultralytics import YOLO
from pathlib import Path
import argparse
import json

# Default configuration
DEFAULT_MODEL = "sublot_segmentation/yolov8_seg/weights/best.pt"
DEFAULT_DATA = "data.yaml"
DEFAULT_SPLIT = "val"
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH = 16
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.6

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Validation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to trained model weights")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="Path to data.yaml")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT,
                        choices=["val", "test", "train"],
                        help="Dataset split to validate on")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE,
                        help="Validation image size")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="Batch size")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help="IoU threshold for NMS")
    parser.add_argument("--save-json", action="store_true",
                        help="Save results to JSON file")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("YOLOv8 Segmentation Validation")
    print("=" * 60)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Error: Model not found at {model_path}")
        print(f"\nPlease train the model first or specify correct path:")
        print(f"  python validate.py --model path/to/weights.pt")
        return
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # STRICT YOLO-SEG ENFORCEMENT
    if model.task != 'segment':
        print(f"ERROR: Model '{args.model}' is a '{model.task}' model.")
        print("This script is strictly for YOLO-SEG (segmentation) models.")
        return
    
    # Check data.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ Error: data.yaml not found at {data_path}")
        return
    
    print(f"Dataset config: {data_path}")
    print(f"Split: {args.split}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch}")
    print("=" * 60)
    
    # Run validation
    print(f"\nRunning validation on {args.split} set...\n")
    
    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.img_size,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=args.save_json,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    
    # Print metrics
    print(f"\nBox Metrics:")
    print(f"  Precision: {results.box.p[0]:.4f}")
    print(f"  Recall:    {results.box.r[0]:.4f}")
    print(f"  mAP50:     {results.box.map50:.4f}")
    print(f"  mAP50-95:  {results.box.map:.4f}")
    
    print(f"\nMask Metrics:")
    print(f"  Precision: {results.seg.p[0]:.4f}")
    print(f"  Recall:    {results.seg.r[0]:.4f}")
    print(f"  mAP50:     {results.seg.map50:.4f}")
    print(f"  mAP50-95:  {results.seg.map:.4f}")
    
    print(f"\nSpeed:")
    print(f"  Preprocess:  {results.speed['preprocess']:.2f} ms")
    print(f"  Inference:   {results.speed['inference']:.2f} ms")
    print(f"  Postprocess: {results.speed['postprocess']:.2f} ms")
    
    # Save results to JSON if requested
    if args.save_json:
        output_file = model_path.parent.parent / f"validation_{args.split}.json"
        
        results_dict = {
            "model": str(model_path),
            "split": args.split,
            "box_metrics": {
                "precision": float(results.box.p[0]),
                "recall": float(results.box.r[0]),
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
            },
            "mask_metrics": {
                "precision": float(results.seg.p[0]),
                "recall": float(results.seg.r[0]),
                "map50": float(results.seg.map50),
                "map50_95": float(results.seg.map),
            },
            "speed": {
                "preprocess_ms": results.speed['preprocess'],
                "inference_ms": results.speed['inference'],
                "postprocess_ms": results.speed['postprocess'],
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
