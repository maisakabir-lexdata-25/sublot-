import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import sys

# Fix for PyTorch 2.6+ security restriction on model loading
try:
    from ultralytics.nn.tasks import SegmentationModel, DetectionModel
    import torch.nn as nn
    # Add project root to sys.path to allow importing from backend
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.utils.post_process import (
        detect_roads, subtract_roads, get_tiles_metadata, 
        is_centroid_in_center, split_merged_mask_watershed, 
        is_rectangular, non_max_suppression_shapely, is_on_border,
        refine_mask_to_content, is_in_corner
    )
    
    # List of classes to allow for secure loading
    safe_classes = [
        SegmentationModel, 
        DetectionModel,
        nn.Sequential,
        nn.ModuleList,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.SiLU,
        nn.Upsample,
        nn.MaxPool2d
    ]
    
    # Add ultralytics specific modules if possible
    try:
        from ultralytics.nn.modules import Conv, C2f, C3, SPPF, Bottleneck, Concat
        safe_classes.extend([Conv, C2f, C3, SPPF, Bottleneck, Concat])
    except ImportError:
        pass
        
    torch.serialization.add_safe_globals(safe_classes)
except Exception:
    pass

# -------------------------------------------------------------------
# Page Config & Styles
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Sublot Intelligence Suite",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Orbitron:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0e1117;
    }
    .main {
        background: radial-gradient(circle at top right, #1a1c24, #0e1117);
        padding: 2rem;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Model Helper Functions
# -------------------------------------------------------------------

def get_mask_rcnn_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

@st.cache_resource
def load_yolo(path):
    if not os.path.exists(path): return None
    model = YOLO(path)
    if model.task != 'segment': return None
    return model

@st.cache_resource
def load_mrcnn(path, device):
    if not os.path.exists(path): return None
    model = get_mask_rcnn_model(2)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except:
        return None

def random_color():
    return tuple(np.random.randint(50, 255, 3).tolist())

# -------------------------------------------------------------------
# Main App Logic
# -------------------------------------------------------------------

st.title("🌿 Sublot Intelligence Pro")
st.markdown("### Advanced Agricultural Instance Segmentation Suite")

# Sidebar Configuration
st.sidebar.header("🕹️ Engine Control")
engine_choice = st.sidebar.radio(
    "Select Model Engine",
    ["YOLOv11-SEG", "Mask R-CNN", "Ultimate Hybrid"],
    index=1, # Default to Mask R-CNN as user is using it
    help="Choose the detection backbone for your analysis."
)

device_choice = st.sidebar.selectbox("Compute Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
DEVICE = torch.device(device_choice)

st.sidebar.markdown("---")
st.sidebar.header("📁 Configuration")

# Dynamic Paths based on Engine
if engine_choice == "YOLOv11-SEG":
    DEFAULT_YOLO_PATH = "backend/models/yolov11_sublot/yolov11m_final/weights/best.pt"
    model_path = st.sidebar.text_input("YOLO Weights (.pt)", value=DEFAULT_YOLO_PATH)
elif engine_choice == "Mask R-CNN":
    DEFAULT_MRCNN_PATH = "mask_rcnn_sublot_20e.pth"
    model_path = st.sidebar.text_input("Mask R-CNN Weights (.pth)", value=DEFAULT_MRCNN_PATH)
else:
    # Hybrid Mode
    DEFAULT_YOLO_PATH = "ultimate_hybrid/yolo_stage/weights/best.pt"
    DEFAULT_MRCNN_PATH = "ultimate_hybrid_refiner.pth"
    yolo_p = st.sidebar.text_input("YOLO Stage Path", value=DEFAULT_YOLO_PATH)
    mrcnn_p = st.sidebar.text_input("Refiner MRCNN Path", value=DEFAULT_MRCNN_PATH)
    model_path = yolo_p # Just for status check

# Advanced Settings
with st.sidebar.expander("🛠️ Advanced Settings", expanded=True):
    conf_threshold = st.slider("Global Sensitivity", 0.01, 1.0, 0.10)
    
    use_tiling = st.checkbox("Enable Intelligent Tiling", value=True)
    use_watershed = st.checkbox("✨ Precision Watershed Snapping", value=True, help="Snaps boundaries exactly to visual edges between fields.")
    
    min_area = st.slider("Min Area (px)", 10, 5000, 120)
    filter_border = st.checkbox("Strict Edge Cleanup", value=False)
    
    detect_surrounding = st.checkbox("🌐 Detect Surrounding Areas", value=True, help="Ultra-aggressive detection for edge and border sublots.")
    
    show_street_debug = st.checkbox("🔍 Show Detected Street Lines", value=False, help="Visualize the detected roads and paths for debugging.")
    show_comparison = st.checkbox("🔄 Show Before/After Comparison", value=False, help="Compare detection with and without street line suppression.")

    
    st.markdown("---")
    st.markdown("### 🎨 Annotation Style")
    use_street_suppression = st.checkbox("🛣️ Street Line Suppression", value=False, help="Uses detected roads and straight field lines to clip and bound sublots.")
    box_scale = st.slider("Boundary Offset", 0.8, 1.1, 1.0)
    smooth_poly = st.slider("Boundary Smoothing", 0, 10, 3, help="Smooths out jagged edges for a professional look.")
    fill_opacity = st.slider("Transparency", 0.0, 1.0, 0.3)
    border_px = st.slider("Border Thickness", 1, 10, 2)

# Load Model
model = None
with st.spinner(f"Waking up {engine_choice} engine..."):
    if engine_choice == "YOLOv11-SEG":
        model = load_yolo(model_path)
    elif engine_choice == "Mask R-CNN":
        model = load_mrcnn(model_path, DEVICE)
    elif engine_choice == "Ultimate Hybrid":
        yolo_h = load_yolo(yolo_p)
        mrcnn_h = load_mrcnn(mrcnn_p, DEVICE)
        model = (yolo_h, mrcnn_h) if yolo_h and mrcnn_h else None

if model is None:
    st.error(f"❌ Failed to load {engine_choice} weights. Please verify the path.")
else:
    st.sidebar.success(f"✅ {engine_choice} Active")

# Main Interface
uploaded_file = st.file_uploader("📤 Upload Satellite/Field Imagery", type=['jpg', 'jpeg', 'png', 'tiff'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h_img, w_img = img_np.shape[:2]
    
    # --- ENGINE 0: PRE-PROCESSING (Contrast Enhancement) ---
    enable_clahe = st.sidebar.checkbox("✨ Enhance Edge Contrast (CLAHE)", value=False, help="Improves visibility in faint/shadowed surroundings.")
    force_unified = st.sidebar.toggle("🧬 Force Unified Fields", value=True, help="Prevents sublots from being split into many pieces by internal lines (Good for purple marks).")
    if enable_clahe:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img_np_proc = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        img_np_proc = img_np.copy()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Input")
        st.image(image, use_container_width=True)
    
    with st.spinner(f"Analyzing {engine_choice} for surroundings..."):
        
        # Function to run detection pipeline
        def run_detection_pipeline(apply_street_suppression):
            """Run the full detection pipeline with or without street line suppression"""
            candidates_local = []
            
            # Tiled Inference Strategy (Maximum overlap for complete edge field coverage)
            if use_tiling:
                tiles_meta = get_tiles_metadata(w_img, h_img, tile_size=640, overlap=0.45)
            else:
                tiles_meta = [(0, 0, w_img, h_img)]

            # Process image for surrounding area detection
            # Add padding if surrounding detection is enabled to catch edge fields better
            if detect_surrounding:
                # Add 48px padding on all sides for better corner/edge context
                img_np_padded = cv2.copyMakeBorder(img_np_proc, 48, 48, 48, 48, cv2.BORDER_REFLECT)
                pad_offset = 48
            else:
                img_np_padded = img_np_proc
                pad_offset = 0

            for tx, ty, tw, th in tiles_meta:
                # Adjust tile coordinates for padding
                tile_img = img_np_padded[ty+pad_offset:ty+th+pad_offset, tx+pad_offset:tx+tw+pad_offset]
                t_image = Image.fromarray(tile_img)
                
                # --- MODEL INFERENCE ---
                if "YOLO" in engine_choice or engine_choice == "Ultimate Hybrid":
                    y_mod = model[0] if engine_choice == "Ultimate Hybrid" else model
                    y_res = y_mod.predict(source=t_image, conf=conf_threshold, verbose=False)[0]
                    if y_res.masks is not None:
                        for idx, p in enumerate(y_res.masks.xy):
                            poly = (np.array(p, dtype=np.int32) + np.array([tx, ty])).astype(np.int32)
                            candidates_local.append({'poly': poly, 'conf': y_res.boxes.conf[idx].item()})

                if "Mask R-CNN" in engine_choice or engine_choice == "Ultimate Hybrid":
                    m_mod = model[1] if engine_choice == "Ultimate Hybrid" else model
                    t_tensor = F.to_tensor(t_image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        preds = m_mod(t_tensor)
                    
                    p_masks = preds[0]['masks'].cpu().numpy()
                    p_scores = preds[0]['scores'].cpu().numpy()
                    for i in range(len(p_scores)):
                        if p_scores[i] < conf_threshold: continue
                        mask_tile = (p_masks[i, 0] > 0.5).astype(np.uint8)
                        cnts, _ = cv2.findContours(mask_tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in cnts:
                            poly = (cnt.reshape(-1, 2) + np.array([tx, ty])).astype(np.int32)
                            candidates_local.append({'poly': poly, 'conf': p_scores[i]})

            # --- PHASE 2: GLOBAL ORCHESTRATION ---
            # 1. High-precision hard NMS using Shapely to remove clear duplicates
            # Use a slightly higher threshold (0.5) to avoid merging adjacent fields
            candidates_local = non_max_suppression_shapely(candidates_local, overlap_thresh=0.5)
            
            candidates_local.sort(key=lambda x: x['conf'], reverse=True)
            polygons_local = []
            occupied_mask_local = np.zeros((h_img, w_img), dtype=np.uint8)
            
            # Global Edge Knowledge
            gray = cv2.cvtColor(img_np_proc, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 150)
            edge_kernel = np.ones((3,3), np.uint8)
            edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
            
            # Pre-detect Roads and Street Lines if enabled
            road_mask_local = detect_roads(img_np_proc) if apply_street_suppression else np.zeros((h_img, w_img), dtype=bool)

            for cand in candidates_local:
                # 2. SHAPE-BASED GHOSTING FILTER (Solve Red Marks)
                # Detect long, thin "ribbon" artifacts that overlap real fields
                p_area = cv2.contourArea(cand['poly'])
                p_perimeter = cv2.arcLength(cand['poly'], True)
                if p_perimeter == 0: continue
                compactness = (4 * np.pi * p_area) / (p_perimeter ** 2)
                
                # If sublot is EXTREMELY thin/jagged (ghosting), drop it if it overlaps a lot
                # Lowered from 0.15 to 0.10 to be more conservative - only filter true artifacts
                is_ghost_candidate = compactness < 0.10

                temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [cand['poly']], 255)
                
                # ROAD/STREET SUBTRACTION - Apply the street line logic
                if apply_street_suppression:
                    temp_mask[road_mask_local] = 0
                
                # ZERO OVERLAP SUBTRACTION
                intersection = cv2.bitwise_and(temp_mask, occupied_mask_local)
                overlap_area = cv2.countNonZero(intersection)
                
                # If a ghost candidate overlaps significantly with already detected fields, IGNORE IT
                # BUT: Always preserve border/corner fields even if they look like ghosts
                if is_ghost_candidate and (overlap_area > p_area * 0.2):
                    # Check if this is a border or corner field - if so, DON'T filter it
                    temp_check = np.zeros((h_img, w_img), dtype=np.uint8)
                    cv2.fillPoly(temp_check, [cand['poly']], 255)
                    is_border_check = is_on_border(temp_check)
                    is_corner_check = is_in_corner(temp_check) if detect_surrounding else False
                    if not (is_border_check or is_corner_check):
                        continue

                temp_mask[occupied_mask_local > 0] = 0
                
                # Check if this is a border or corner field
                is_border_field = is_on_border(temp_mask)
                is_corner_field = is_in_corner(temp_mask) if detect_surrounding else False
                
                # MULTI-TIER area check: Corner > Border > Interior
                # Apply progressively aggressive thresholds based on location
                if is_corner_field:
                    # EXTREME mode for corners: 3% threshold (hardest to detect)
                    c_min_area = min_area * 0.03
                elif is_border_field:
                    if detect_surrounding:
                        # ULTRA mode for borders: 5% threshold
                        c_min_area = min_area * 0.05
                    else:
                        # Normal border mode: 10% threshold
                        c_min_area = min_area * 0.1
                else:
                    # Standard interior threshold
                    c_min_area = min_area
                    
                if cv2.countNonZero(temp_mask) < c_min_area: continue
                
                if force_unified:
                    # Keep fields whole (Solve purple/blue marks AND striped/textured fields)
                    # First, close internal gaps caused by stripes or texture
                    closing_kernel = np.ones((7,7), np.uint8)
                    temp_mask_closed = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=2)
                    
                    # Then dilate slightly to bridge thin gaps between nearby regions
                    bridge_kernel = np.ones((5,5), np.uint8)
                    temp_mask_closed = cv2.dilate(temp_mask_closed, bridge_kernel, iterations=1)
                    
                    # Now extract unified contours
                    cnts, _ = cv2.findContours(temp_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        if cv2.contourArea(cnt) >= c_min_area:
                            polygons_local.append(cnt.reshape(-1, 2))
                            cv2.fillPoly(occupied_mask_local, [cnt], 255)
                else:
                    # Precision Watershed logic for 'Perfect' boundaries
                    if use_watershed:
                        # Create marker for watershed
                        # Background is already marked as 'occupied'
                        # We create local markers inside the temp_mask
                        dist_transform = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
                        _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
                        sure_fg = np.uint8(sure_fg)
                        
                        cnts, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in cnts:
                            if cv2.contourArea(cnt) >= c_min_area:
                                polygons_local.append(cnt.reshape(-1, 2))
                                cv2.fillPoly(occupied_mask_local, [cnt], 255)
                    else:
                        # Standard Splitting logic
                        split_mask = temp_mask.copy()
                        split_mask[edges_dilated > 0] = 0
                        num_labels, labels = cv2.connectedComponents(split_mask)
                        for label in range(1, num_labels):
                            comp = (labels == label).astype(np.uint8) * 255
                            if cv2.countNonZero(comp) >= c_min_area:
                                comp = cv2.dilate(comp, edge_kernel, iterations=1)
                                comp = cv2.bitwise_and(comp, temp_mask)
                                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in cnts:
                                    if cv2.contourArea(cnt) >= c_min_area:
                                        polygons_local.append(cnt.reshape(-1, 2))
                                        cv2.fillPoly(occupied_mask_local, [cnt], 255)

            # Smart Border logic
            if filter_border:
                new_f_polys = []
                for p in polygons_local:
                    area = cv2.contourArea(p)
                    if area > 300: new_f_polys.append(p)
                    else:
                        on_e = False
                        for pt in p:
                            if pt[0] <= 1 or pt[1] <= 1 or pt[0] >= w_img-1 or pt[1] >= h_img-1:
                                on_e = True; break
                        if not on_e: new_f_polys.append(p)
                polygons_local = new_f_polys
            
            return polygons_local, road_mask_local
        
        # Run detection with current settings
        polygons, road_mask = run_detection_pipeline(use_street_suppression)
        
        # If comparison mode is enabled, also run without street suppression
        if show_comparison:
            polygons_before, _ = run_detection_pipeline(False)
        

    with col2:
        st.subheader(f"🎯 Detection Result ({len(polygons)})")
        
        if polygons:
            canvas = img_np.copy()
            overlay = img_np.copy()
            colors = [(79, 172, 254), (0, 242, 254), (74, 194, 154), (180, 236, 81), (247, 249, 151), (238, 156, 167)]
            
            for idx, poly in enumerate(polygons):
                color = colors[idx % len(colors)]
                color_bgr = (color[2], color[1], color[0])
                
                # REFINEMENT & SMOOTHING
                if smooth_poly > 0:
                    epsilon = 0.0005 * smooth_poly * cv2.arcLength(poly, True)
                    poly = cv2.approxPolyDP(poly, epsilon, True)
                
                # We no longer use convexHull here because it causes overlaps on adjacent fields.
                # Instead, we use the smoothed polygon directly.
                target_poly = poly
                
                M = cv2.moments(target_poly)
                if M["m00"] != 0 and box_scale != 1.0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    refined = ((target_poly - [cX, cY]) * box_scale + [cX, cY]).astype(np.int32)
                else: 
                    refined = target_poly
                
                cv2.fillPoly(overlay, [refined], color_bgr)
                cv2.polylines(canvas, [refined], True, color_bgr, border_px)
            
            cv2.addWeighted(overlay, fill_opacity, canvas, 1-fill_opacity, 0, canvas)
            st.image(canvas, use_container_width=True)
            
            buf = io.BytesIO()
            Image.fromarray(canvas).save(buf, format="PNG")
            st.download_button("⬇️ Download Results", data=buf.getvalue(), file_name=f"sublots_perfect.png", mime="image/png")
        else:
            st.warning("No sublots detected. Try lowering Global Sensitivity.")
            st.image(image, use_container_width=True)
    
    # Street Line Debug Visualization
    if show_street_debug and use_street_suppression:
        st.markdown("---")
        st.subheader("🔍 Street Line Detection Debug")
        
        debug_canvas = img_np.copy()
        # Overlay the detected road mask in cyan
        road_overlay = np.zeros_like(img_np.copy())
        road_overlay[road_mask] = [0, 255, 255]  # Cyan color
        cv2.addWeighted(road_overlay, 0.5, debug_canvas, 1.0, 0, debug_canvas)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Detected Street/Road Mask (Cyan)**")
            st.image(debug_canvas, use_container_width=True)
        
        with col_d2:
            st.markdown("**Pure Street Mask**")
            # Show the mask in black and white
            street_viz = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            street_viz[road_mask] = 255
            st.image(street_viz, use_container_width=True)
    
    # Before/After Comparison Visualization
    if show_comparison:
        st.markdown("---")
        st.subheader("🔄 Before/After Comparison")
        st.info(f"**Before** (No Street Suppression): {len(polygons_before)} sublots | **After** (With Street Suppression): {len(polygons)} sublots")
        
        # Helper function to render polygons
        def render_polygons(poly_list, img_base):
            canvas = img_base.copy()
            overlay = img_base.copy()
            colors = [(79, 172, 254), (0, 242, 254), (74, 194, 154), (180, 236, 81), (247, 249, 151), (238, 156, 167)]
            
            for idx, poly in enumerate(poly_list):
                color = colors[idx % len(colors)]
                color_bgr = (color[2], color[1], color[0])
                
                # REFINEMENT & SMOOTHING
                if smooth_poly > 0:
                    epsilon = 0.0005 * smooth_poly * cv2.arcLength(poly, True)
                    poly = cv2.approxPolyDP(poly, epsilon, True)
                
                target_poly = poly
                
                M = cv2.moments(target_poly)
                if M["m00"] != 0 and box_scale != 1.0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    refined = ((target_poly - [cX, cY]) * box_scale + [cX, cY]).astype(np.int32)
                else: 
                    refined = target_poly
                
                cv2.fillPoly(overlay, [refined], color_bgr)
                cv2.polylines(canvas, [refined], True, color_bgr, border_px)
            
            cv2.addWeighted(overlay, fill_opacity, canvas, 1-fill_opacity, 0, canvas)
            return canvas
        
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.markdown("**🔴 BEFORE: Without Street Lines**")
            before_canvas = render_polygons(polygons_before, img_np)
            st.image(before_canvas, use_container_width=True)
        
        with col_comp2:
            st.markdown("**🟢 AFTER: With Street Line Suppression**")
            after_canvas = render_polygons(polygons, img_np)
            st.image(after_canvas, use_container_width=True)



# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; opacity: 0.5;'>Sublot Intelligence Suite v4.5 | Optimized for Edge Detection</div>", unsafe_allow_html=True)
