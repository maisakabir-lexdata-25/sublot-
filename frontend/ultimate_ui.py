
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
from ultralytics import YOLO, FastSAM, SAM

# Set page configuration
st.set_page_config(
    page_title="Ultimate Hybrid Detection (YOLO+MaskRCNN+SAM)",
    page_icon="🦅",
    layout="wide"
)

# Dark Futuristic Premium CSS
st.markdown("""
    <style>
    .main {
        background-color: #050508;
        color: #ffffff;
    }
    .stSlider > div > div > div > div { background-color: #00f2ff; }
    h1 {
        background: linear-gradient(90deg, #00f2ff, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }
    .metric-card {
        background: rgba(20, 20, 30, 0.8);
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.1);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#050508, #101020);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Model Loading Logic
# -------------------------------------------------------------------

def get_maskrcnn_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

@st.cache_resource
def load_all_models(yolo_p, mask_p, device):
    try:
        # 1. YOLO Stage
        yolo = YOLO(yolo_p)
        
        # 2. Mask R-CNN Stage
        mask_m = get_maskrcnn_model(2)
        mask_m.load_state_dict(torch.load(mask_p, map_location=device))
        mask_m.to(device)
        mask_m.eval()
        
        # 3. FULL SAM 2 Stage (The "Entire" Model)
        # We use sam2_l (Large) for maximum boundary accuracy
        sam_engine = SAM("sam2_l.pt") 
        
        return yolo, mask_m, sam_engine
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None

# -------------------------------------------------------------------
# Ultimate Hybrid Orchestrator
# -------------------------------------------------------------------

def random_color():
    return tuple(np.random.randint(100, 255, 3).tolist())

def process_ultimate(image, yolo, mask_m, sam_engine, threshold, alpha, thickness, use_sam=True, buffer_size=0):
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    # --- PHASE 1: YOLO PROPOSALS (Extreme Mode) ---
    yolo_results = yolo.predict(
        img_np, 
        conf=threshold * 0.5, 
        imgsz=640, 
        verbose=False, 
        augment=True,
        max_det=300
    )
    
    # --- PHASE 2: MASK R-CNN REFINEMENT ---
    tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_results = mask_m(tensor)
    
    mr_boxes = mask_results[0]['boxes'].cpu().numpy()
    mr_scores = mask_results[0]['scores'].cpu().numpy()
    
    # --- PHASE 3: INTELLIGENT ENSEMBLE ---
    all_proposals = []
    
    def is_on_edge(box):
        x1, y1, x2, y2 = box
        pad = 20
        return x1 < pad or y1 < pad or x2 > (w - pad) or y2 > (h - pad)

    # 1. Collect Mask R-CNN Proposals
    for i in range(len(mr_scores)):
        score = mr_scores[i]
        adj_threshold = threshold * 0.7 if is_on_edge(mr_boxes[i]) else threshold
        if score >= adj_threshold:
            all_proposals.append({'box': mr_boxes[i], 'score': score, 'type': 'mrcnn'})
            
    # 2. Collect YOLO Proposals
    if yolo_results[0].boxes:
        y_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        y_scores = yolo_results[0].boxes.conf.cpu().numpy()
        for i in range(len(y_scores)):
            score = y_scores[i]
            adj_threshold = threshold * 0.7 if is_on_edge(y_boxes[i]) else threshold
            if score >= adj_threshold:
                all_proposals.append({'box': y_boxes[i], 'score': score, 'type': 'yolo'})

    # Priority Sort
    all_proposals = sorted(all_proposals, key=lambda x: x['score'] + (0.2 if is_on_edge(x['box']) else 0), reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    result_img = img_np.copy()
    count = 0

    # Initialize SAM 2 Predictor for the entire image
    # Note: Using 'sam_engine' as a predictor call
    for prop in all_proposals:
        box = prop['box']
        
        # Grow box slightly for better SAM context
        bx1, by1, bx2, by2 = box
        bx1, by1 = max(0, bx1-5), max(0, by1-5)
        bx2, by2 = min(w, bx2+5), min(h, by2+5)
        refined_box = [bx1, by1, bx2, by2]

        mask = None
        # --- PHASE 4: FULL SAM 2 SEGMENTATION ("The Entire Model") ---
        if use_sam:
            # Full SAM 2 Box Prompting
            # sam_engine.predict provides the high-fidelity mask from the Entire SAM 2 model
            results = sam_engine.predict(img_np, bboxes=[refined_box], device=DEVICE, verbose=False)
            if results and len(results[0].masks) > 0:
                mask = results[0].masks.data[0].cpu().numpy().astype(bool)
        
        if mask is None: continue
        
        # Apply Buffer/Dilation if requested
        if buffer_size > 0:
            kernel = np.ones((buffer_size, buffer_size), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Trim & Occupied Logic
        mask_area = np.sum(mask)
        if mask_area < 20: continue
        
        intersection = np.sum(mask & occupied)
        if intersection / mask_area > 0.6: continue
        
        clean_mask = mask & ~occupied
        if np.sum(clean_mask) < 20: continue
        
        occupied |= clean_mask
        color = random_color()
        count += 1
        
        # Visualization
        c_layer = np.zeros_like(img_np)
        c_layer[clean_mask] = color
        result_img[clean_mask] = (result_img[clean_mask] * (1-alpha) + c_layer[clean_mask] * alpha).astype(np.uint8)
        
        # Draw smooth boundary
        cnts, _ = cv2.findContours(clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, cnts, -1, color, thickness)

    return result_img, count

# -------------------------------------------------------------------
# App Interface
# -------------------------------------------------------------------

st.write("# 🦅 Ultimate Hybrid Sublot Analysis")
st.markdown("### YOLOv11 proposals + Mask R-CNN refinement + SAM boundary snapping")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
YOLO_PATH = "ultimate_hybrid/yolo_stage/weights/best.pt"
MASK_PATH = "ultimate_hybrid_refiner.pth"

yolo, mask_m, sam = load_all_models(YOLO_PATH, MASK_PATH, DEVICE)

if yolo and mask_m:
    st.sidebar.markdown("## 🕹️ AI Core Control")
    conf = st.sidebar.slider("Global Sensitivity", 0.01, 0.9, 0.25, help="Lower this to 0.01 to detect small/edge sublots.")
    
    st.sidebar.info("💡 Tip: For edge/distant sublots, set Sensitivity to 0.10 - 0.20")
    
    sam_enabled = st.sidebar.toggle("🌌 SAM Boundary Snapping (sam3)", value=True)
    boundary_buffer = st.sidebar.slider("Boundary Buffer (Dilation)", 0, 10, 2)
    mask_alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.4)
    border_px = st.sidebar.slider("Border Width", 1, 5, 2)

    file = st.sidebar.file_uploader("Upload Imagery", type=['jpg','png','jpeg'])
    
    if file:
        img = Image.open(file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Original Multispectral/RGB Input", use_container_width=True)
            
        with st.spinner("Executing Triple-Stage Inference..."):
            res, det_count = process_ultimate(img, yolo, mask_m, sam, conf, mask_alpha, border_px, sam_enabled, boundary_buffer)
            
        with col2:
            st.image(res, caption=f"Ultimate Result: {det_count} Sublots Detected", use_container_width=True)
            
        st.success(f"Detections finalized: {det_count} sublots found.")
else:
    st.warning("⚠️ High-performance models are missing. Please ensure Phase 2 weights are ready.")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Proprietary Hybrid Engine v3.0</div>", unsafe_allow_html=True)
