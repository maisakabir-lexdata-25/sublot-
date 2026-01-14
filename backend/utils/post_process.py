import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

def detect_roads(image_np, brightness_threshold=160, edge_threshold1=30, edge_threshold2=100):
    """
    Detect roads, paths, and 'street lines' in the image.
    Agricultural paths are often neutral (grey/brown) or brighter than fields.
    Uses a combination of color thresholding and Hough Line detection for straight paths.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Detect bright roads (concrete/paved)
    bright_mask = (gray > brightness_threshold).astype(np.uint8) * 255
    
    # 2. Detect neutral paths (dirt roads often have low saturation)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    path_mask = ((saturation < 40) & (gray > 80)).astype(np.uint8) * 255
    
    # 3. Use Edge detection
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
    kernel = np.ones((5, 5), np.uint8)
    
    # 4. Hough Line Detection for 'Street Lines' (LESS AGGRESSIVE)
    # Agricultural fields are bounded by very straight paths.
    street_mask = np.zeros_like(gray)
    # Increased threshold to 120 (from 80) and minLineLength to 100 (from 50) for fewer false positives
    # Reduced line thickness to 3 (from 5) to not remove too much field area
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=100, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw lines with less thickness to avoid removing field content
            cv2.line(street_mask, (x1, y1), (x2, y2), 255, 3)
    
    # Combined Road/Street Mask - Make it less aggressive
    # Only use bright roads and the Hough lines, skip the low-saturation mask which can be too broad
    combined_road = cv2.bitwise_or(bright_mask, street_mask)
    
    # Don't dilate edges as much to preserve field boundaries
    edges_dilated = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    # Only add edges if they coincide with detected roads
    edges_on_roads = cv2.bitwise_and(edges_dilated, combined_road)
    combined_road = cv2.bitwise_or(combined_road, edges_on_roads)
    
    # Less aggressive morphology
    combined_road = cv2.morphologyEx(combined_road, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    return combined_road > 0

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

def is_in_corner(mask_np, corner_margin=0.15):
    """Check if mask is in a corner region (within 15% of edges from corners)."""
    h, w = mask_np.shape
    margin_h = int(h * corner_margin)
    margin_w = int(w * corner_margin)
    
    # Check if mask has pixels in any corner region
    top_left = mask_np[:margin_h, :margin_w].any()
    top_right = mask_np[:margin_h, w-margin_w:].any()
    bottom_left = mask_np[h-margin_h:, :margin_w].any()
    bottom_right = mask_np[h-margin_h:, w-margin_w:].any()
    
    return top_left or top_right or bottom_left or bottom_right

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




def refine_mask_to_content(mask_np, image_np):
    """
    Refines a mask to stay 'inside' the border of the field content.
    Uses local color and edge data to prune pixels that look like dirt/roads.
    """
    if mask_np.sum() < 100:
        return mask_np
        
    # Create a mask of the content
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # The refined mask: Keep original mask pixels that are NOT on a strong edge 
    # (edges often define the boundary of the field)
    refined = mask_np.copy()
    refined[edges_dilated > 0] = 0
    
    # Clean up shards
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    
    # If we lost too much, fall back to a slightly eroded version of original
    if refined.sum() < mask_np.sum() * 0.5:
        return cv2.erode(mask_np, kernel, iterations=1)
        
    return refined

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
