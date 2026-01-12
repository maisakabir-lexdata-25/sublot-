import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

def test_inference():
    # Load a small SegFormer model (unlabeled) to verify logic
    # In practice, we'd use 'segformer_sublot/final_model'
    model_id = "nvidia/mit-b0"
    
    print(f"Loading processor and model: {model_id}")
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id, num_labels=2)
    
    # Create dummy image
    image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    print("Running inference...")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    print(f"Prediction shape: {prediction.shape}")
    print("✓ Inference logic verified!")

if __name__ == "__main__":
    test_inference()
