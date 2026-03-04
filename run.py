import cv2
import numpy as np
import onnxruntime as ort
from utilities import *
import logging
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

cap = cv2.VideoCapture("demovid/vid1.avi")  # or path to video
if not cap.isOpened():
    raise FileNotFoundError("Video file not found")

# ===== CONFIG =====
fps = cap.get(cv2.CAP_PROP_FPS)
FPS_VIDEO = fps if fps > 0 else 30 # frame rate of the input video (default to 30 if unknown)
TOTAL_TIME_DETECT = 2.5               # Detection window duration (seconds)
FRAME_PER_DETECT  = 8                 # Number of classifier inputs per window
DETECT_INTERVAL   = int(FPS_VIDEO * TOTAL_TIME_DETECT / FRAME_PER_DETECT)  # Frames between each detection (e.g., every 10 frames for 30 FPS)

logger.info(f"Video detected: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {FPS_VIDEO}fps")

# Tracking parameters
TRACKER      = "MOSSE"                # Tracker type (e.g., "KCF", "MOSSE", "MEDIANFLOW")
MAX_TRACKS   = 5                      # Maximum simultaneous tracks
CONF_ON      = 0.25                   # Confidence threshold to show track
CONF_OFF     = 0.1                    # Confidence threshold to hide/remove track
k            = 5.0                    # Distance penalty multiplier (higher = prefer existing tracks)
STICK_WEIGHT = 0.7                    # Weight of stickiness in scoring
alpha        = 0.8                    # EMA smoothing factor for box coordinates (0.8 = 80% old, 20% new)
TRACKER_FAILURE_DECAY = 0.5           # Confidence decay factor when tracker fails (0.5 = halve confidence)

COLORS = [
    (0,   0,   255),
    (0,   255, 0  ),
    (255, 0,   0  ),
    (0,   255, 255),
    (255, 0,   255),
]


# Load ONNX model
def load_onnx_models()->tuple[ort.InferenceSession, ort.InferenceSession]:
    """Load ONNX models with proper error handling."""
    model_paths = {
        'yolo': 'violence_yolo.onnx',
        'gap': 'gapconv1d.onnx'
    }
    
    try:
        yolo_session = ort.InferenceSession(
            model_paths['yolo'],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        gap_session = ort.InferenceSession(
            model_paths['gap'],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        return yolo_session, gap_session
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to load ONNX models: {e}")
        exit(1)

def name_onnx_model(YOLO_session, GAPConv1D_session)->tuple[str, list[str], str, str]:
    yolo_input_name = YOLO_session.get_inputs()[0].name
    yolo_output_names = [o.name for o in YOLO_session.get_outputs()]

    gap_input_name = GAPConv1D_session.get_inputs()[0].name
    gap_output_name = GAPConv1D_session.get_outputs()[0].name
    return yolo_input_name,yolo_output_names,gap_input_name,gap_output_name

def Detect(frame, YOLO_session, yolo_input_name, yolo_output_names)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run YOLO detection and return rescaled boxes, feature vector, and areas."""
    H, W = frame.shape[:2]
    # Resize to model size
    img, scale, pad_x, pad_y = letterbox_image(frame, (320, 320))

    # Convert uint8 → float32 with proper shape
    img_input = img.astype(np.float32) / 255.0  # uint8 to float32
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
    img_input = np.expand_dims(img_input, 0)  # Add batch dimension

    yolo_outputs = YOLO_session.run(
        yolo_output_names, 
        {yolo_input_name: img_input}
    )

    detections = yolo_outputs[0]     # (1, 5, 6)
    feature    = yolo_outputs[-1]     # (1, 512)

    detections = np.squeeze(detections, axis=0)
    feature    = np.squeeze(feature, axis=0)
    
    
    boxes = rescale_boxes(detections, scale, pad_x, pad_y, (H, W))  # rescale to original frame size
    boxes, areas = merge_overlapping_boxes(boxes)  # merge overlapping detections into one (optional, can help reduce noise)
    
    return boxes, feature, areas

def score_track(max_tracks, distance_penalty, stickiness_weight, tracks, frame_height, frame_width, boxes, areas)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    confs = boxes[:, 4]  # Extract confidence scores from the last column
    boxes = boxes[:, :4]  # Extract box coordinates

    if tracks:
        prev     = np.array([t['box'] for t in tracks])
        dist_mat = l1_center_dist_matrix(prev, boxes) / (frame_width + frame_height)
        best_stick = np.clip(1.0 - distance_penalty * dist_mat.min(axis=0), 0.0, 1.0)
    else:
        best_stick = np.zeros(len(boxes))

    scores  = confs * areas * (1.0 + stickiness_weight * best_stick)
    top_idx = np.argsort(scores)[::-1][:max_tracks]
    return boxes,confs,top_idx

YOLO_session, GAPConv1D_session = load_onnx_models()
yolo_input_name, yolo_output_names, gap_input_name, gap_output_name = name_onnx_model(YOLO_session, GAPConv1D_session)

# Initialization
frame_id   = 0
tracks    = []
raw_boxes = []
raw_confs = []
feats     = deque(maxlen=FRAME_PER_DETECT)  # store last N features for classifier input


def tracking(tracker, tracks, frame, det_boxes, det_confs, i):
    tr = make_tracker(tracker)
    tr.init(frame, xyxy_to_xywh(det_boxes[i]))
    tracks.append({
                        'box':        det_boxes[i],
                        'conf':       det_confs[i],
                        'show':       False,
                        'tracker':    tr,
                        'tracker_ok': True,
                    })

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    is_detect_frame = (frame_id % DETECT_INTERVAL == 0)
    is_classifier_frame = (frame_id % DETECT_INTERVAL == 5)

    # ===== BETWEEN-FRAME TRACKING =====
    if not is_detect_frame:
        for t in tracks:
            if t['tracker'] is None or not t['tracker_ok']:
                # tracker failed or not yet initialized → hold position
                continue

            ok, new_xywh = t['tracker'].update(frame)
            if ok:
                new_box = xywh_to_xyxy(new_xywh)
                # sanity check: box must stay within frame
                if (new_box[0] >= 0 and new_box[1] >= 0 and
                    new_box[2] <= W  and new_box[3] <= H  and
                    new_box[2] > new_box[0] and new_box[3] > new_box[1]):
                    t['box'] = alpha * t['box'] + (1 - alpha) * new_box
                    t['tracker_ok'] = True
                else:
                    t['tracker_ok'] = False  # out-of-bounds → treat as failed
            else:
                t['tracker_ok'] = False  # tracker lost target
                t['conf'] *= TRACKER_FAILURE_DECAY # decay confidence if tracker fails

    # ===== YOLO DETECTION =====
    if is_detect_frame:
        boxes, feature, areas = Detect(frame, YOLO_session, yolo_input_name, yolo_output_names)
        feats.append(feature)

        raw_boxes = []
        raw_confs = []

        if len(boxes) > 0:

            boxes, confs, top_idx = score_track(MAX_TRACKS, k, STICK_WEIGHT, tracks, H, W, boxes, areas)
            raw_boxes = boxes[top_idx]
            raw_confs = confs[top_idx]

        # ===== MATCH YOLO → TRACKS =====
        if len(raw_boxes) > 0:
            det_boxes = np.array(raw_boxes)
            det_confs = np.array(raw_confs)

            if not tracks:
                for i in range(len(det_boxes)):
                    tracking(TRACKER, tracks, frame, det_boxes, det_confs, i)
                    tracks = merge_overlapping_tracks(tracks) 
            else:
                prev     = np.array([t['box'] for t in tracks])
                dist_mat = l1_center_dist_matrix(prev, det_boxes) / (W + H)

                matched_det = set()
                matched_trk = set()

                flat_order = np.argsort(dist_mat, axis=None)
                for idx in flat_order:
                    ti, di = divmod(int(idx), dist_mat.shape[1])
                    if ti in matched_trk or di in matched_det:
                        continue
                    # Re-init tracker on every YOLO frame for matched tracks
                    # (corrects any tracker drift)
                    tr = make_tracker(TRACKER)
                    tr.init(frame, xyxy_to_xywh(det_boxes[di]))
                    tracks[ti]['box']        = alpha * tracks[ti]['box'] + (1-alpha) * det_boxes[di]
                    tracks[ti]['conf']       = alpha * tracks[ti]['conf'] + (1-alpha) * det_confs[di]
                    tracks[ti]['tracker']    = tr
                    tracks[ti]['tracker_ok'] = True
                    matched_trk.add(ti)
                    matched_det.add(di)
                    if len(matched_trk) == min(len(tracks), len(det_boxes)):
                        break

                # Unmatched detections → new tracks
                for di in range(len(det_boxes)):
                    if di not in matched_det and len(tracks) < MAX_TRACKS:
                        tracking(TRACKER, tracks, frame, det_boxes, det_confs, di)

                # Unmatched tracks → decay
                for ti in range(len(tracks)):
                    if ti not in matched_trk:
                        tracks[ti]['conf']       *= TRACKER_FAILURE_DECAY
                        tracks[ti]['tracker_ok']  = False  # don't trust tracker either

        else:
            # YOLO found nothing → decay all tracks
            for t in tracks:
                t['conf']       *= TRACKER_FAILURE_DECAY
                t['tracker_ok']  = False

        # Prune dead tracks
        tracks = [t for t in tracks if t['conf'] >= CONF_OFF]
        tracks = merge_overlapping_tracks(tracks) 

    # ===== CLASSIFIER INFERENCE =====
    if is_classifier_frame and len(feats) >= 8:
        # Run classifier on the last 8 features
        seq = np.stack(list(feats), axis=0)  # ✅ Convert deque to list
        seq = np.expand_dims(seq, 0)    # (1, 8, 512)
        gap_output = GAPConv1D_session.run(
            [gap_output_name],
            {gap_input_name: seq}
        )

        logits = gap_output[0]  # shape: (1, 1)

        probs = 1.0 / (1.0 + np.exp(-logits))

        violence_prob = probs[0][0]
        logger.info(f"Frame {frame_id}: Violence probability = {violence_prob:.4f} "
                f"(Active tracks: {len(tracks)})")
    
        # Optional: Alert on high confidence
        if violence_prob > 0.8:
            logger.warning(f"HIGH VIOLENCE DETECTED!")
            cv2.putText(frame, f"⚠ VIOLENCE: {violence_prob:.2f}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                        (0, 0, 255), 3)

    # ===== HYSTERESIS =====
    for t in tracks:
        if t['conf'] >= CONF_ON:
            t['show'] = True
        elif t['conf'] < CONF_OFF:
            t['show'] = False

    # ===== DRAW =====
    for i, t in enumerate(tracks):
        if not t['show']:
            continue
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = map(int, t['box'])
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # Show tracker status in label
        status = "OK" if t['tracker_ok'] else "HOLD"
        cv2.putText(frame, f"#{i} {t['conf']:.2f} [{status}]", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()