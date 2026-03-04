import cv2
import numpy as np
from typing import Tuple, Optional

def letterbox_image(
    img: np.ndarray, 
    inp_dim: Tuple[int, int], 
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize image with unchanged aspect ratio using padding.
    
    Args:
        img: Input image (H, W, 3)
        inp_dim: Target dimensions (width, height)
        pad_color: Padding color (BGR format for OpenCV)
    
    Returns:
        canvas: Padded resized image
        scale: Scaling factor applied
        pad_x: X-axis padding offset
        pad_y: Y-axis padding offset
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")
    
    img_h, img_w = img.shape[:2]
    w, h = inp_dim
    
    if w <= 0 or h <= 0:
        raise ValueError("Input dimensions must be positive")

    scale = min(w / img_w, h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    # Resize with aspect ratio preserved
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas with padding
    canvas = np.full((h, w, 3), pad_color, dtype=np.uint8)

    # Calculate padding to center the image
    pad_x = (w - new_w) // 2
    pad_y = (h - new_h) // 2

    # Place resized image on canvas
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return canvas, scale, pad_x, pad_y

def rescale_boxes(
    boxes: np.ndarray, 
    scale: float, 
    pad_x: int, 
    pad_y: int, 
    original_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Rescale bounding boxes from letterboxed space back to original image space.
    Preserves confidence scores, class IDs, and other metadata.
    
    Args:
        boxes: Nx4+ array of boxes in letterboxed space
            Format: (x1, y1, x2, y2, confidence, class_id, ...)
            First 4 columns are coordinates (required)
            Remaining columns are preserved as-is (confidence, class, etc.)
        scale: Scaling factor used in letterbox_image
        pad_x: X-axis padding offset from letterbox_image
        pad_y: Y-axis padding offset from letterbox_image
        original_shape: Original image shape (height, width)
    
    Returns:
        rescaled_boxes: Boxes with coordinates rescaled, metadata preserved
            Shape: (N, M) where M >= 4 (same as input)
    
    Raises:
        ValueError: If boxes array is invalid
    
    Example:
        >>> boxes = np.array([[100, 150, 200, 250, 0.95, 0]], dtype=np.float32)
        >>> rescaled = rescale_boxes(boxes, 0.5, 10, 20, (480, 640))
        >>> print(rescaled)
        # [[180. 260. 380. 480.  0.95  0. ]]  # coords rescaled, conf & class preserved
    """
    
    # Handle None or empty
    if boxes is None or boxes.size == 0:
        return np.array([], dtype=np.float32).reshape(0, 4)
    
    # Convert to float32 and copy
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    
    # Handle 1D array (single detection)
    if boxes.ndim == 1:
        if len(boxes) < 4:
            raise ValueError(
                f"Boxes must have at least 4 values (x1, y1, x2, y2), got {len(boxes)}"
            )
        boxes = boxes.reshape(1, -1)
    
    # Validate shape
    if boxes.ndim != 2:
        raise ValueError(f"Boxes must be 1D or 2D array, got shape {boxes.shape}")
    
    if boxes.shape[1] < 4:
        raise ValueError(
            f"Boxes must have at least 4 columns (x1, y1, x2, y2), got shape {boxes.shape}"
        )
    
    if scale <= 0:
        raise ValueError(f"Scale factor must be positive, got {scale}")
    
    # Split coordinates from metadata
    coords = boxes[:, :4].copy()
    metadata = boxes[:, 4:] if boxes.shape[1] > 4 else np.array([])
    
    # Step 1: Remove padding from coordinates
    coords[:, 0] -= pad_x  # x1
    coords[:, 2] -= pad_x  # x2
    coords[:, 1] -= pad_y  # y1
    coords[:, 3] -= pad_y  # y2
    
    # Step 2: Undo scaling
    coords /= scale
    
    # Step 3: Clip to original image bounds
    h, w = original_shape
    coords[:, 0] = np.clip(coords[:, 0], 0, w)  # x1
    coords[:, 2] = np.clip(coords[:, 2], 0, w)  # x2
    coords[:, 1] = np.clip(coords[:, 1], 0, h)  # y1
    coords[:, 3] = np.clip(coords[:, 3], 0, h)  # y2
    
    # Combine rescaled coordinates with original metadata
    if metadata.size > 0:
        result = np.hstack([coords, metadata])
    else:
        result = coords
    
    return result

def make_tracker(type: str = "KCF") -> cv2.Tracker:
    if type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    elif type == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()
    elif type == "KCF":
        return cv2.legacy.TrackerKCF_create()
    raise ValueError(f"Unknown tracker: {type}")

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)

def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)

def l1_center_dist_matrix(tracked_boxes, detected_boxes):
    def centers(b):
        return np.stack([(b[:,0]+b[:,2])*0.5, (b[:,1]+b[:,3])*0.5], axis=1)
    tc = centers(tracked_boxes)
    dc = centers(detected_boxes)
    diff = np.abs(tc[:,None,:] - dc[None,:,:])
    return diff.sum(axis=2)

def merge_overlapping_tracks(tracks, iou_threshold=0.45):
    """
    Merge pairs of tracks whose boxes overlap above iou_threshold.
    Keeps the higher-conf track, discards the other.
    Runs repeatedly until no more merges needed.
    """
    if len(tracks) <= 1:
        return tracks

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)

    changed = True
    while changed:
        changed = False
        keep = [True] * len(tracks)
        for i in range(len(tracks)):
            if not keep[i]:
                continue
            for j in range(i+1, len(tracks)):
                if not keep[j]:
                    continue
                if iou(tracks[i]['box'], tracks[j]['box']) >= iou_threshold:
                    # keep higher conf, absorb the other's conf slightly
                    winner, loser = (i, j) if tracks[i]['conf'] >= tracks[j]['conf'] else (j, i)
                    tracks[winner]['conf'] = max(tracks[winner]['conf'], tracks[loser]['conf'])
                    keep[loser] = False
                    changed = True
        tracks = [t for i, t in enumerate(tracks) if keep[i]]

    return tracks

def merge_overlapping_boxes(boxes, iou_threshold=0.45):
    """
    Merge pairs of boxes whose IoU exceeds iou_threshold.
    Keeps the box with higher confidence, discards the other.
    Runs repeatedly until no more merges needed.
    
    Args:
        boxes: NxM array of boxes, where first 4 columns are (x1, y1, x2, y2) and remaining are metadata (confidence, class_id, etc.)
        iou_threshold: IoU threshold for merging boxes
    
    Returns:
        merged_boxes: Array of boxes after merging overlaps
    """
    if len(boxes) <= 1:
        return boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid_mask = areas >= 100  # Keep boxes with area >= 100
    boxes = boxes[valid_mask]
    areas = areas[valid_mask]

    def iou(a, b, area_a, area_b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        return inter / (area_a + area_b - inter)

    changed = True
    while changed:
        changed = False
        keep = [True] * len(boxes)
        for i in range(len(boxes)):
            if not keep[i]:
                continue
            for j in range(i+1, len(boxes)):
                if not keep[j]:
                    continue
                if iou(boxes[i][:4], boxes[j][:4], areas[i], areas[j]) >= iou_threshold:
                    # keep higher conf box
                    winner, loser = (i, j) if boxes[i][4] >= boxes[j][4] else (j, i)
                    keep[loser] = False
                    changed = True
        boxes = np.array([b for i, b in enumerate(boxes) if keep[i]])
        areas = np.array([a for i, a in enumerate(areas) if keep[i]])

    return boxes, areas