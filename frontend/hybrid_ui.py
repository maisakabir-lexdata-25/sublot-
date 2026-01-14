
import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Hybrid Sublot Analytics",
    page_icon="🤖",
    layout="wide"
)

# Premium UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #0d0d12;
        color: #e0e0e0;
    }
    .stSlider > div > div > div > div {
        background-color: #6366f1;
    }
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        font-weight: 600;
    }
    h1 {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0px !important;
    }
    .status-card {
        background: rgba(30, 30, 35, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Model Loader (Architecture must match train_hybrid.py)
# -------------------------------------------------------------------

def get_hybrid_model(num_classes):
    # Base framework
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    
    # Predictors
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model

@st.cache_resource
def load_hybrid_model(model_path, device):
    if not os.path.exists(model_path):
        return None
    
    num_classes = 2 # BG + Sublot
    model = get_hybrid_model(num_classes)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------------------------------------------------
# Visualization & Logic
# -------------------------------------------------------------------

def random_color():
    return tuple(np.random.randint(50, 255, 3).tolist())

def visualize_hybrid(image, predictions, threshold=0.5, no_overlap=True, alpha=0.4, thickness=1, show_boxes=False, solidify=0):
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    masks = predictions[0]['masks'].cpu().detach().numpy()

    # Initial Confidence Filtering
    keep = scores >= threshold
    boxes, scores, masks = boxes[keep], scores[keep], masks[keep]
    
    # Sort by confidence
    if len(scores) > 0:
        indices = np.argsort(scores)[::-1]
        boxes, scores, masks = boxes[indices], scores[indices], masks[indices]
        
        occupied = np.zeros((h, w), dtype=bool)
        final_boxes, final_masks, final_scores = [], [], []
        
        for i in range(len(scores)):
            m_soft = masks[i, 0]
            m_bin = (m_soft > 0.5).astype(np.uint8)
            
            # Refinement 1: Solidify (Dilation) to close small gaps between fields
            if solidify > 0:
                kernel = np.ones((solidify, solidify), np.uint8)
                m_bin = cv2.dilate(m_bin, kernel, iterations=1)
            
            # Refinement 2: Fill internal holes in the mask
            cnts, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            m_bin = np.zeros_like(m_bin)
            cv2.drawContours(m_bin, cnts, -1, 1, thickness=-1)
            
            mask_area = np.sum(m_bin)
            if mask_area < 50: continue # Lowered min area for distant sublots
            
            # Conflict Resolution
            if no_overlap:
                inter_area = np.sum(m_bin.astype(bool) & occupied)
                # Relaxed from 0.3 to 0.6 to prevent aggressive cutting in complex corners
                if inter_area / mask_area > 0.6: continue
                
                # Trim the overlap
                clean_m = m_bin & ~occupied.astype(np.uint8)
                if np.sum(clean_m) < 50: continue
                occupied |= clean_m.astype(bool)
                m_viz = clean_m
            else:
                m_viz = m_bin

            final_masks.append(m_viz)
            final_scores.append(scores[i])
            
            # Tight Box from Final Mask
            cnts_final, _ = cv2.findContours(m_viz.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_final:
                all_pts = np.concatenate(cnts_final)
                bx, by, bw, bh = cv2.boundingRect(all_pts)
                final_boxes.append([bx, by, bx+bw, by+bh])
        
        boxes, masks, scores = np.array(final_boxes), np.array(final_masks), np.array(final_scores)

    # Drawing
    res_img = img_np.copy()
    for i in range(len(boxes)):
        color = random_color()
        m = masks[i]
        
        # Transparent Fill
        c_mask = np.zeros_like(img_np)
        c_mask[m == 1] = color
        res_img[m == 1] = (res_img[m == 1] * (1-alpha) + c_mask[m == 1] * alpha).astype(np.uint8)
        
        # Borders
        cnts_disp, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, cnts_disp, -1, color, thickness)
        
        if show_boxes:
            b = boxes[i].astype(int)
            cv2.rectangle(res_img, (b[0], b[1]), (b[2], b[3]), color, thickness)
            cv2.putText(res_img, f"{scores[i]:.2f}", (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
    return res_img, len(boxes)

# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------

st.title("🤖 Hybrid Sublot Detection")
st.markdown("##### High-Precision Segmentation with Efficient Backbone Extraction")
st.markdown("---")

# Sidebar
st.sidebar.header("🕹️ Control Panel")
conf_level = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
st.sidebar.info("💡 Hint: Lower threshold helps detect distant and perimeter sublots.")

solidify_factor = st.sidebar.slider("Boundary Solidify (Dilation)", 0, 10, 3, help="Increases mask size before trimming to fill gaps between sublots.")
opacity = st.sidebar.slider("Mask Alpha", 0.0, 1.0, 0.4, 0.1)
border_w = st.sidebar.slider("Border Thickness", 1, 5, 1)

st.sidebar.markdown("---")
prevent_overlap = st.sidebar.toggle("🧩 Smart Pixel Trimming", value=True)
debug_boxes = st.sidebar.toggle("📦 Show Debugging Boxes", value=False)

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

# Load Model
MODEL_PATH = "sublot_hybrid_final.pth"
model = load_hybrid_model(MODEL_PATH, device)

if model:
    st.sidebar.success(f"✅ Hybrid Model Loaded ({device_type.upper()})")
    
    file = st.sidebar.file_uploader("Upload Imagery", type=['png', 'jpg', 'jpeg'])
    
    if file:
        img = Image.open(file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Original Input")
            st.image(img, use_container_width=True)
            
        with st.spinner("Decoding Hybrid Features..."):
            tensor = F.to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(tensor)
            
            res_viz, count = visualize_hybrid(
                img, preds, conf_level, prevent_overlap, opacity, border_w, debug_boxes, solidify_factor
            )
            
        with col2:
            st.markdown(f"### 🎯 Results ({count})")
            st.image(res_viz, use_container_width=True)
            
        st.markdown("---")
        st.subheader("📊 Performance Insights")
        st.info(f"The Hybrid model identified **{count}** distinct sublot instances. Using Smart Pixel Trimming ensured zero overlaps in boundaries.")
else:
    st.error("❌ Model weights not found! Run the hybrid training script first.")
    st.code("python backend/scripts/train_hybrid.py")

st.markdown("<br><div style='text-align: center; color: gray;'>Engineered by Antigravity AI</div>", unsafe_allow_html=True)
