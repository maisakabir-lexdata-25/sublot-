import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #ffffff;
    }
    .main {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    }
    .stMetric {
        background: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #dcfce7;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4CAF50;
    }
    .stSidebar {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }
    /* Glassmorphism card effect */
    .css-1r6slb0, .css-12oz5g7 {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(76, 175, 80, 0.2);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌿 Sublot Intelligence Pro")
st.markdown("### Precision Instance Segmentation for Agricultural Fields")

# Sidebar for model upload
st.sidebar.header("📁 Model Configuration")

model_path = st.sidebar.text_input("Model Path", value="yolov11_sublot/yolov11m_final/weights/best.pt")

with st.sidebar.expander("🛠️ Advanced Settings", expanded=False):
    conf_threshold = st.slider("Confidence", 0.05, 0.99, 0.10, help="Lower this if sublots are not being detected.")
    iou_threshold = st.slider("IOU Threshold", 0.1, 0.9, 0.45)
    min_area = st.slider("Min Area (px)", 100, 5000, 100)
    rectangularity_threshold = st.slider("Rectangularity", 0.0, 1.0, 0.3, help="How rectangular the shape must be (1.0 = perfect rectangle)")
    filter_border = st.checkbox("Only Complete Sublots", value=True)

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

import os
# Auto-load model on startup
if not st.session_state.model_loaded:
    try:
        # Check both possible paths
        p1 = "sublot/yolov11_sublot/yolov11m_final/weights/best.pt"
        p2 = "yolov11_sublot/yolov11m_final/weights/best.pt"
        target_p = p1 if os.path.exists(p1) else p2
        
        if os.path.exists(target_p):
            st.session_state.model = YOLO(target_p)
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
            st.image(image, use_container_width=True)
        
        # Analysis execution
        # Auto-run if image is uploaded (Activation On)
        if True: # Always process if we reach here and have an image
            with st.spinner("Processing image with YOLOv11-SEG..."):
                try:
                    # Helper function to check if shape is rectangular
                    def is_rectangular(poly, threshold=0.5):
                        """Check if polygon is rectangular based on geometry."""
                        # Calculate rectangularity (area / bounding box area)
                        x, y, w, h = cv2.boundingRect(poly)
                        bbox_area = w * h
                        if bbox_area == 0:
                            return False
                        
                        poly_area = cv2.contourArea(poly)
                        rectangularity = poly_area / bbox_area
                        
                        # Check aspect ratio (not too skinny)
                        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                        if aspect_ratio > 15:  # Very permissive - allows long thin fields
                            return False
                        
                        return rectangularity >= threshold
                    
                    # Run YOLO inference
                    results = st.session_state.model.predict(
                        source=image,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False,
                        device='cpu'
                    )[0]
                    
                    from shapely.geometry import Polygon
                    from shapely.validation import make_valid
                    
                    # Store candidates with confidence
                    candidates = []
                    
                    # Debug counters
                    debug_counts = {
                        'raw_detections': 0,
                        'after_area_filter': 0,
                        'after_rect_filter': 0,
                        'after_border_filter': 0
                    }
                    
                    if results.masks is not None:
                        h_img, w_img = image.size[::-1]
                        
                        # Get boxes for confidence scores
                        boxes = results.boxes
                        
                        debug_counts['raw_detections'] = len(results.masks.xy)
                        
                        for idx, mask_data in enumerate(results.masks.xy):
                            # mask_data is already in polygon format
                            poly = np.array(mask_data, dtype=np.int32)
                            
                            # Area filtering
                            area = cv2.contourArea(poly)
                            if area < min_area:
                                continue
                            debug_counts['after_area_filter'] += 1
                            
                            # Rectangle shape validation
                            if not is_rectangular(poly, rectangularity_threshold):
                                continue
                            debug_counts['after_rect_filter'] += 1
                            
                            # Border filtering (Complete sublots only)
                            if filter_border:
                                on_border = False
                                for pt in poly:
                                    x, y = pt
                                    if x <= 2 or y <= 2 or x >= w_img - 2 or y >= h_img - 2:
                                        on_border = True
                                        break
                                if on_border:
                                    continue
                            debug_counts['after_border_filter'] += 1
                            
                            # Get confidence (default to 0 if not available)
                            conf = boxes.conf[idx].item() if boxes is not None else 0.0
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
                            
                        # Convert numpy to list of tuples for Shapely
                        try:
                            shapely_poly = Polygon(poly_np)
                            if not shapely_poly.is_valid:
                                shapely_poly = make_valid(shapely_poly)
                        except Exception:
                            continue
                            
                        # Calculate available area (Difference)
                        try:
                            # If it's the first one, it's fully accepted
                            if occupied_union.is_empty:
                                clean_poly = shapely_poly
                            else:
                                clean_poly = shapely_poly.difference(occupied_union)
                            
                            # If fully overlapped, skip
                            if clean_poly.is_empty:
                                continue
                                
                            # If result is MultiPolygon (fragmented), take the largest chunk
                            if clean_poly.geom_type == 'MultiPolygon':
                                clean_poly = max(clean_poly.geoms, key=lambda a: a.area)
                            
                            if clean_poly.geom_type != 'Polygon':
                                continue
                                
                            # Check if remaining area is still significant (> min_area)
                            if clean_poly.area < min_area:
                                continue
                                
                            # Add to final list
                            # Convert back to numpy
                            x, y = clean_poly.exterior.coords.xy
                            final_np = np.array([list(zip(x, y))], dtype=np.int32).reshape(-1, 2)
                            polygons.append(final_np)
                            
                            # Update occupied area
                            occupied_union = occupied_union.union(clean_poly)
                            
                        except Exception as e:
                            # Fallback if geometry math fails
                            print(f"Geometry error: {e}")
                            pass
                    
                    num_sublots = len(polygons)
                    
                    # Display results
                    with col2:
                        st.subheader("🎯 Identification Results")
                        
                    
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
                            
                            st.image(img_np, use_container_width=True, caption=f"Identified {num_sublots} instances")
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
                                use_container_width=True
                            )
                        else:
                            st.image(image, use_container_width=True)
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
