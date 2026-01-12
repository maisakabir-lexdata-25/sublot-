import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

def detect_roads(image_np, brightness_threshold=180, edge_threshold1=50, edge_threshold2=150):
    """
    Detect roads using brightness and Canny edge detection.
    Roads are typically light-colored with clear edges in agricultural satellite imagery.
    """
    # Convert to grayscale
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 3: # RGB/BGR
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np[:,:,0]
    else:
        gray = image_np
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)
    
    # Detect bright pixels
    bright_mask = (gray > brightness_threshold).astype(np.uint8) * 255
    
    # Combine edges with brightness
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    road_mask = cv2.bitwise_and(bright_mask, edges_dilated)
    
    # Final dilation to ensure road coverage
    road_mask = cv2.dilate(road_mask, kernel, iterations=1)
    
    return road_mask > 0

def subtract_roads(mask_np, road_mask, min_component_area=100):
    """
    Subtract road pixels from a binary mask and separate into connected components.
    """
    mask_no_roads = mask_np.copy()
    mask_no_roads[road_mask] = 0
    
    num_labels, labels = cv2.connectedComponents(mask_no_roads)
    components = []
    
    for label in range(1, num_labels):
        comp = (labels == label).astype(np.uint8) * 255
        if np.sum(comp > 0) > min_component_area:
            components.append(comp)
            
    return components

def split_merged_mask_watershed(mask_np, peak_threshold=0.3, min_area=50):
    """
    Splits one giant mask into multiple subplots using Distance Transform and Watershed.
    """
    if mask_np.max() == 0:
        return []

    # Distance Transform
    dist_transform = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
    
    # Threshold to find peaks (centers of subplots)
    ret, last_peak = cv2.threshold(dist_transform, peak_threshold * dist_transform.max(), 255, 0)
    peaks = np.uint8(last_peak)
    
    # Find markers
    ret, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[mask_np == 0] = 0
    
    # Watershed
    img_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_3ch, markers)
    
    split_polys = []
    for label in np.unique(markers):
        if label <= 1: # Skip background/boundaries
            continue
            
        subplot_mask = np.zeros_like(mask_np, dtype=np.uint8)
        subplot_mask[markers == label] = 255
        
        contours, _ = cv2.findContours(subplot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                split_polys.append(cnt.reshape(-1, 2))
                
    return split_polys

def is_on_border(mask_np):
    """Check if mask touches its own frame border."""
    return (
        mask_np[0, :].any() or mask_np[-1, :].any() or 
        mask_np[:, 0].any() or mask_np[:, -1].any()
    )

def get_tiles_metadata(img_w, img_h, tile_size=640, overlap=0.25):
    """Calculate tile coordinates with overlap."""
    tiles = []
    stride = int(tile_size * (1 - overlap))
    
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x1, y1 = x, y
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            
            # Shift back if tile is clipped to maintain size
            if x2 == img_w and x1 > 0:
                x1 = max(0, img_w - tile_size)
            if y2 == img_h and y1 > 0:
                y1 = max(0, img_h - tile_size)
            
            tiles.append((x1, y1, x2 - x1, y2 - y1))
            
            if x + tile_size >= img_w:
                break
        if y + tile_size >= img_h:
            break
    return tiles

def is_centroid_in_center(poly, tile_rect, overlap, img_size):
    """Check if polygon centroid lies in the 'safe' center region of the tile."""
    tx, ty, tw, th = tile_rect
    IW, IH = img_size
    
    M = cv2.moments(poly)
    if M["m00"] == 0:
        return False
        
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    
    mx = (tw * overlap) / 2
    my = (th * overlap) / 2
    
    left = tx + mx if tx > 0 else 0
    right = tx + tw - mx if tx + tw < IW else IW
    top = ty + my if ty > 0 else 0
    bottom = ty + th - my if ty + th < IH else IH
    
    return left <= cX <= right and top <= cY <= bottom

def is_rectangular(poly, threshold=0.3):
    """Check if polygon is rectangular based on bounding box extent."""
    x, y, w, h = cv2.boundingRect(poly)
    bbox_area = w * h
    if bbox_area == 0: 
        return False
    poly_area = cv2.contourArea(poly)
    rectangularity = poly_area / bbox_area
    return rectangularity >= threshold

def non_max_suppression_shapely(candidates, overlap_thresh=0.3):
    """
    Perform Non-Maximum Suppression using Shapely polygons for high precision.
    candidates: list of {'poly': np.array, 'conf': float}
    """
    # Sort by confidence
    candidates.sort(key=lambda x: x['conf'], reverse=True)
    
    accepted = []
    for cand in candidates:
        poly_np = cand['poly']
        if len(poly_np) < 3:
            continue
            
        try:
            shapely_poly = Polygon(poly_np)
            if not shapely_poly.is_valid:
                shapely_poly = make_valid(shapely_poly)
        except Exception:
            continue
            
        is_duplicate = False
        for acc in accepted:
            try:
                acc_poly = Polygon(acc['poly'])
                if not acc_poly.is_valid:
                    acc_poly = make_valid(acc_poly)
                    
                intersection_area = shapely_poly.intersection(acc_poly).area
                if intersection_area > shapely_poly.area * overlap_thresh or \
                   intersection_area > acc_poly.area * overlap_thresh:
                    is_duplicate = True
                    break
            except Exception:
                continue
                
        if not is_duplicate:
            accepted.append(cand)
            
    return accepted
