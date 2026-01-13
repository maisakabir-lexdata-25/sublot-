
import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Mask R-CNN Sublot Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        border: none;
        color: white;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(75, 108, 183, 0.4);
    }
    .uploadedFile {
        background-color: #262730;
        border-radius: 8px;
        padding: 1rem;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .css-1d391kg {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Description
st.title("🧬 Mask R-CNN Sublot Detection")
st.markdown("### Advanced Instance Segmentation for Field Sublots")
st.markdown("---")

# -------------------------------------------------------------------
# Model Definition (Must match training script exactly)
# -------------------------------------------------------------------

def get_mask_rcnn_model(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None) # Weights=None because we load custom weights

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

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

@st.cache_resource
def load_model(model_path, device):
    if not os.path.exists(model_path):
        return None
    
    num_classes = 2 # Background + Sublot
    model = get_mask_rcnn_model(num_classes)
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def visualize_prediction(image, predictions, threshold=0.5, no_overlap=False, show_boxes=False, alpha=0.5, thickness=1):
    # Convert PIL to Numpy
    img_np = np.array(image)
    
    # Masks and Boxes
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    masks = predictions[0]['masks'].cpu().detach().numpy()

    # Filter by threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]
    
    # Sort by score descending (highest confidence first)
    if no_overlap and len(scores) > 0:
        sorted_indices = np.argsort(scores)[::-1]
        boxes = boxes[sorted_indices]
        masks = masks[sorted_indices]
        scores = scores[sorted_indices]
        
        occupied_mask = np.zeros(img_np.shape[:2], dtype=bool)
        
        cleaned_boxes = []
        cleaned_masks = []
        cleaned_scores = []
        
        for i in range(len(scores)):
            mask = masks[i, 0] > 0.5  # Soft mask to binary
            mask_area = np.sum(mask)
            
            if mask_area == 0:
                continue
                
            # Check overlap with higher confidence masks
            intersection = mask & occupied_mask
            intersection_area = np.sum(intersection)
            overlap_ratio = intersection_area / mask_area
            
            if overlap_ratio > 0.3:
                continue

            clean_mask = mask & ~occupied_mask
            
            if np.sum(clean_mask) > 100:  
                occupied_mask |= clean_mask  
                
                contours, _ = cv2.findContours(clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    for c in contours[1:]:
                        nx, ny, nw, nh = cv2.boundingRect(c)
                        x = min(x, nx)
                        y = min(y, ny)
                        w = max(x+w, nx+nw) - x
                        h = max(y+h, ny+nh) - y
                    
                    cleaned_boxes.append([x, y, x+w, y+h])
                    cleaned_masks.append(clean_mask.astype(np.uint8)) 
                    cleaned_scores.append(scores[i])
        
        if len(cleaned_boxes) > 0:
            boxes = np.array(cleaned_boxes)
            masks = np.array(cleaned_masks) 
            scores = np.array(cleaned_scores)
        else:
            boxes = np.empty((0, 4))
            masks = np.empty((0, 1, 1))
            scores = np.array([])

    result_img = img_np.copy()
    
    # Overlay Masks
    for i in range(len(boxes)):
        color = random_color()
        box = boxes[i].astype(int)
        
        if no_overlap:
             mask = masks[i]
        else:
             mask = (masks[i, 0] > 0.5).astype(np.uint8)
        
        # Draw Mask
        colored_mask = np.zeros_like(img_np)
        colored_mask[mask == 1] = color
        
        # Blend
        mask_indices = mask == 1
        result_img[mask_indices] = (result_img[mask_indices] * (1 - alpha) + \
                                   colored_mask[mask_indices] * alpha).astype(np.uint8)
        
        # Draw Border (Thin lines)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, color, thickness)
        
        # Optional Bounding Box and Score
        if show_boxes:
            cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), color, thickness)
            text = f"{scores[i]:.2f}"
            cv2.putText(result_img, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return result_img, len(boxes)

# -------------------------------------------------------------------
# Main App Logic
# -------------------------------------------------------------------

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
st.sidebar.info("💡 Tip: Lower the confidence threshold to detect more sublots on the edges.")

mask_opacity = st.sidebar.slider("Mask Opacity (Alpha)", 0.0, 1.0, 0.4, 0.1)
line_thickness = st.sidebar.slider("Line Thickness", 1, 5, 1)

no_overlap_mode = st.sidebar.checkbox("⛔ Prevent Overlapping", value=True)
show_boxes_mode = st.sidebar.checkbox("🔲 Show Bounding Boxes", value=False)
device_option = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
device = torch.device(device_option)

# Model Path (Assuming root directory based on training script)
MODEL_PATH = "mask_rcnn_sublot_20e.pth"

# Load Model
model = load_model(MODEL_PATH, device)

if model is None:
    st.warning("⚠️ Model file not found. Please ensure `mask_rcnn_sublot_20e.pth` exists in the project root.")
    st.info("💡 If training is currently running, please wait for it to finish.")
else:
    st.sidebar.success("✅ Model Loaded Successfully")

    # File Uploader
    uploaded_file = st.sidebar.file_uploader("Upload Field Image", type=['jpg', 'jpeg', 'png'])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Run Inference
        with st.spinner("Running Inference..."):
            img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(img_tensor)

            # Visualize
            result_img, count = visualize_prediction(
                image, 
                predictions, 
                confidence_threshold, 
                no_overlap=no_overlap_mode,
                show_boxes=show_boxes_mode
            )
            
        with col2:
            st.subheader(f"Detections ({count})")
            st.image(result_img, use_container_width=True)
            
        # Metrics / Info
        st.markdown("### Analysis Report")
        st.info(f"Detected **{count}** potential sublots with confidence > {confidence_threshold}")
            
    else:
        st.write("👈 Upload an image from the sidebar to get started.")
        
        # Placeholder / Demo
        st.markdown("""
        #### Instructions
        1. Wait for training to complete.
        2. Upload a field image.
        3. Adjust the confidence threshold to filter predictions.
        4. View detected sublots and segments.
        """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with ❤️ by Antigravity</div>", unsafe_allow_html=True)
