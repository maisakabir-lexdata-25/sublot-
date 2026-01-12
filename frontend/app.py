import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import torch

# Fix for PyTorch 2.6+ security restriction on model loading
try:
    from ultralytics.nn.tasks import SegmentationModel, DetectionModel
    import torch.nn as nn
    import sys
    import os
    # Add project root to sys.path to allow importing from backend
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.utils.post_process import (
        detect_roads, subtract_roads, get_tiles_metadata, 
        is_centroid_in_center, split_merged_mask_watershed, 
        is_rectangular, non_max_suppression_shapely, is_on_border
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

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #12141a; /* Deeper slate but cleaner contrast */
    }
    .main {
        background: linear-gradient(135deg, #161a21 0%, #12141a 100%);
        padding: 2rem;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-left: 4px solid #4CAF50;
    }
    .stSidebar {
        background-color: #1c2128 !important; /* Lighter sidebar for better text visibility */
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Improve text contrast */
    h1, h2, h3, p, label, .stMarkdown {
        color: #e6edf3 !important;
    }
    /* Glassmorphism card effect */
    .css-1r6slb0, .css-12oz5g7 {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌿 Sublot Intelligence Pro")
st.markdown("### Precision Instance Segmentation for Agricultural Fields")

import os

# Sidebar for model upload
st.sidebar.header("📁 Model Configuration")

# Check common locations for the model
default_path = "backend/models/yolov11_sublot/yolov11m_final/weights/best.pt"
if not os.path.exists(default_path):
    # Try alternate location if running from within frontend/
    alt_path = "../backend/models/yolov11_sublot/yolov11m_final/weights/best.pt"
    if os.path.exists(alt_path):
        default_path = alt_path

model_path = st.sidebar.text_input("Model Path", value=default_path)

with st.sidebar.expander("🛠️ Advanced Settings", expanded=True):
    conf_threshold = st.slider("Confidence", 0.01, 1.0, 0.10, help="Lower this if sublots are not being detected.")
    iou_threshold = st.slider("IOU Threshold", 0.1, 0.9, 0.45, help="Standard NMS threshold.")
    min_area = st.slider("Min Area (px)", 10, 10000, 100)
    filter_border = st.checkbox("Only Complete Subplots", value=True, help="Remove sublots touching the image edge.")
    enforce_roads = st.checkbox("Enforce Road Boundaries", value=True, help="Detect roads and use them to split merged agricultural fields.")
    use_tiling = st.checkbox("Use Tiled Inference", value=True, help="Process large images in overlapping tiles to avoid boundary artifacts.")
    disable_geometry = st.checkbox("Disable Shape Validation", value=True, help="Keep all shapes regardless of how rectangular they are.")
    split_merged = st.checkbox("Split Merged Sublots (Watershed)", value=True, help="Use distance transform to break giant merged fields into individual subplots.")
    show_raw = st.checkbox("Show Raw YOLO Detections", value=False, help="Debug mode: show everything the model sees.")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Auto-load model on startup
if not st.session_state.model_loaded:
    try:
        if os.path.exists(default_path):
            st.session_state.model = YOLO(default_path)
            if st.session_state.model.task == 'segment':
                st.session_state.model_loaded = True
    except Exception:
        pass

# Load model button (manual override)
if st.sidebar.button("Load/Reload Model"):
    try:
        with st.spinner("Loading YOLOv11-SEG model..."):
            st.session_state.model = YOLO(model_path)
            # Verify it's a segmentation model
            if st.session_state.model.task != 'segment':
                st.sidebar.error(f"❌ Error: This is a '{st.session_state.model.task}' model, not a segmentation model!")
                st.session_state.model_loaded = False
            else:
                st.session_state.model_loaded = True
                st.sidebar.success("✅ YOLOv11-SEG Model loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.session_state.model_loaded = False

# Main content area
if st.session_state.model_loaded:
    st.success("🎯 Model is ready! Upload an image to detect sublots.")
    
    # Image upload
    uploaded_image = st.file_uploader(
        "Upload Field Image",
        type=["jpg", "png", "jpeg", "bmp", "tiff"],
        help="Upload an image of the field to detect sublots"
    )
    
    if uploaded_image:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, width="stretch")
        
        # Analysis execution
        # Auto-run if image is uploaded (Activation On)
        if True: # Always process if we reach here and have an image
            with st.spinner("Processing image with YOLOv11-SEG..."):
                try:
                    # Run YOLO inference Using GPU (RTX 4060)
                    results = st.session_state.model.predict(
                        source=image,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False,
                        device=0 # Using GPU 0
                    )[0]
                    
                    from shapely.geometry import Polygon
                    from shapely.validation import make_valid
                    
                    # Store candidates with confidence
                    candidates = []
                    
                    if results.masks is not None:
                        h_img, w_img = image.size[::-1]
                        
                        # Get boxes for confidence scores
                        boxes = results.boxes
                        
                        # (Functions removed - now imported from backend.utils.post_process)

                        # (Function removed - now imported)

                        raw_results = []
                        img_np = np.array(image)
                        h_img, w_img = img_np.shape[:2]

                        if use_tiling:
                            tiles_meta = get_tiles_metadata(w_img, h_img)
                            for tx, ty, tw, th in tiles_meta:
                                tile_img = img_np[ty:ty+th, tx:tx+tw]
                                t_res = st.session_state.model.predict(
                                    source=tile_img, conf=conf_threshold, iou=iou_threshold, 
                                    verbose=False, device=0
                               )[0]
                                if t_res.masks is not None:
                                    for m, b, p in zip(t_res.masks.data, t_res.boxes.conf, t_res.masks.xy):
                                        global_poly = p + np.array([tx, ty])
                                        if is_centroid_in_center(global_poly, (tx, ty, tw, th), 0.25, (w_img, h_img)):
                                            raw_results.append({'mask': m, 'conf': b.item(), 'offset': (tx, ty), 'size': (tw, th)})
                        else:
                            res = st.session_state.model.predict(
                                source=img_np, conf=conf_threshold, iou=iou_threshold, 
                                verbose=False, device=0
                            )[0]
                            if res.masks is not None:
                                for m, b in zip(res.masks.data, res.boxes.conf):
                                    raw_results.append({'mask': m, 'conf': b.item(), 'offset': (0, 0), 'size': (w_img, h_img)})

                        road_mask = detect_roads(img_np) if enforce_roads else None

                        for item in raw_results:
                            mask_tensor = item['mask']
                            tx, ty = item['offset']
                            tw, th = item['size']
                            conf = item['conf']
                            
                            # Convert to numpy and resize to tile dimensions
                            mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                            if mask_np.shape != (th, tw):
                                mask_np = cv2.resize(mask_np, (tw, th))
                            
                            # Rule 1: Kill boundary predictions
                            if filter_border:
                                is_complete = not (
                                    mask_np[0, :].any() or mask_np[-1, :].any() or 
                                    mask_np[:, 0].any() or mask_np[:, -1].any()
                                )
                                if not is_complete:
                                    continue
                            
                            # Rule 3: Enforce Road Boundaries
                            if enforce_roads and road_mask is not None:
                                # Get full size mask for road subtraction
                                full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                                mh, mw = mask_np.shape
                                full_mask[ty:ty+mh, tx:tx+mw] = mask_np
                                
                                field_masks = subtract_roads(full_mask, road_mask)
                                mask_list = [fm[ty:ty+mh, tx:tx+mw] for fm in field_masks]
                            else:
                                mask_list = [mask_np]

                            for m in mask_list:
                                # Rule 2: Split merged masks
                                if split_merged:
                                    sub_polys = split_merged_mask_watershed(m)
                                    for sp in sub_polys:
                                        global_sp = sp + np.array([tx, ty])
                                        if cv2.contourArea(global_sp) >= min_area:
                                            if disable_geometry or is_rectangular(global_sp):
                                                candidates.append({'poly': global_sp, 'conf': conf})
                                else:
                                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    for cnt in contours:
                                        poly = cnt.reshape(-1, 2) + np.array([tx, ty])
                                        if cv2.contourArea(poly) >= min_area:
                                            if disable_geometry or is_rectangular(poly):
                                                candidates.append({'poly': poly, 'conf': conf})
                    
                    # Sort candidates by confidence (highest first) to prioritize better detections
                    candidates.sort(key=lambda x: x['conf'], reverse=True)
                    
                    # Non-overlapping logic using Shapely
                    polygons = []
                    occupied_union = Polygon()
                    
                    for cand in candidates:
                        poly_np = cand['poly']
                        
                        # Create Shapely polygon (must have >= 3 points)
                        if len(poly_np) < 3:
                            continue
                            
                        try:
                            shapely_poly = Polygon(poly_np)
                            if not shapely_poly.is_valid:
                                shapely_poly = make_valid(shapely_poly)
                        except Exception:
                            continue
                            
                    if not show_raw:
                        candidates = non_max_suppression_shapely(candidates, overlap_thresh=0.3)
                        polygons = [c['poly'] for c in candidates]
                    else:
                        polygons = [c['poly'] for c in candidates]
                    
                    num_sublots = len(polygons)
                    
                    # Display results
                    with col2:
                        st.subheader("🎯 Identification Results")
                        
                        # Add explanation of calculation
                        with st.expander("ℹ️ How was this calculated?"):
                            st.write(f"""
                            **Calculation Logic:**
                            1. **Detection**: YOLOv11 detected **{len(candidates)}** raw potential sublots.
                            2. **Filtering**: We removed objects smaller than {min_area}px. {"(Border filter active)" if filter_border else "(Border filter inactive)"}
                            3. **Duplicate Suppression**: We prioritized high-confidence detections. If a detection overlapped significantly with a better one, it was discarded. Otherwise, the **full original boundary** was preserved.
                            4. **Final Count**: **{num_sublots}** unique sublots identified.
                            """)

                        if num_sublots > 0:
                            # Create visualization
                            img_np = np.array(image)
                            overlay = img_np.copy()
                            
                            # Premium color palette
                            colors = [
                                (76, 175, 80), (139, 195, 74), (205, 220, 57),
                                (255, 235, 59), (255, 193, 7), (255, 152, 0),
                                (255, 87, 34), (233, 30, 99), (156, 39, 176),
                                (103, 58, 183), (63, 81, 181), (33, 150, 243)
                            ]
                            
                            for idx, poly in enumerate(polygons):
                                color = colors[idx % len(colors)]
                                # BGR conversion for OpenCV
                                color_bgr = (color[2], color[1], color[0])
                                
                                # Draw filled polygon with transparency
                                cv2.fillPoly(overlay, [poly], color_bgr)
                                # Draw glowing border
                                cv2.polylines(img_np, [poly], True, (255, 255, 255), 3)
                                cv2.polylines(img_np, [poly], True, color_bgr, 1)
                            
                            # Blend overlay with dynamic opacity
                            cv2.addWeighted(overlay, 0.5, img_np, 0.5, 0, img_np)
                            
                            st.image(img_np, width="stretch", caption=f"Identified {num_sublots} instances")
                            st.success(f"✅ **{num_sublots} Separate Sublots Identified**")
                            
                            # Metrics
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("Total sublots", num_sublots)
                            with metric_col2:
                                total_area = sum(cv2.contourArea(p) for p in polygons)
                                st.metric("Total Area (px)", f"{int(total_area):,}")
                            
                            # Download
                            buf = io.BytesIO()
                            Image.fromarray(img_np).save(buf, format="PNG")
                            st.download_button(
                                label="⬇️ Download YOLO Result",
                                data=buf.getvalue(),
                                file_name="yolo_result.png",
                                mime="image/png",
                                width="stretch"
                            )
                        else:
                            st.image(image, width="stretch")
                            st.warning("⚠️ No sublots identified in this image.")
                            
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")

    else:
        st.info("👆 Please upload an image to begin")

else:
    st.warning("⚠️ Please load your trained YOLOv11-SEG model to begin")
    st.markdown("""
    ### 📝 YOLOv11-SEG Identification Workflow:
    1. **Train**: Run `python train.py` to train your model.
    2. **Load**: Enter the path to your model (.pt file) in the sidebar.
    3. **Detect**: Upload an image to identify sublot polygons.
    
    *Note: YOLOv11-SEG provides native instance segmentation with separate polygons.*
    """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This application uses YOLOv11-SEG model to detect sublots in field images. "
    "Upload your trained model and images to get started."
)
