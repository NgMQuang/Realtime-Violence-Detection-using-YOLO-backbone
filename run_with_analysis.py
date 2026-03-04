"""
VIOLENCE DETECTION SYSTEM WITH INTEGRATED ANALYSIS
Tracks FPS, timing, memory, and bottlenecks in real-time
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import psutil
import os
from collections import deque
from typing import Dict
import sys

sys.path.insert(0, '.')
from run.utilities import *


class QuickAnalyzer:
    """Lightweight analyzer for integrated analysis"""
    
    def __init__(self, window_size: int = 60):
        self.frame_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.tracking_times = deque(maxlen=window_size)
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.frame_count = 0
    
    def record_frame(self, frame_time: float, inference_time: float = 0, tracking_time: float = 0):
        self.frame_times.append(frame_time)
        self.inference_times.append(inference_time)
        self.tracking_times.append(tracking_time)
        self.frame_count += 1
    
    def get_fps(self) -> float:
        if not self.frame_times:
            return 0
        return 1.0 / np.mean(list(self.frame_times))
    
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_avg_times(self) -> Dict[str, float]:
        return {
            'frame': np.mean(list(self.frame_times)) * 1000 if self.frame_times else 0,
            'inference': np.mean(list(self.inference_times)) * 1000 if self.inference_times else 0,
            'tracking': np.mean(list(self.tracking_times)) * 1000 if self.tracking_times else 0,
        }


# ===== CONFIG =====
FPS_VIDEO         = 30
TOTAL_TIME_DETECT = 2.5
FRAME_PER_DETECT  = 8
DETECT_INTERVAL   = int(FPS_VIDEO * TOTAL_TIME_DETECT / FRAME_PER_DETECT)

MAX_TRACKS   = 5
CONF_ON      = 0.25
CONF_OFF     = 0.1
k            = 5.0
STICK_WEIGHT = 0.7
alpha        = 0.8

COLORS = [
    (0,   0,   255),
    (0,   255, 0  ),
    (255, 0,   0  ),
    (0,   255, 255),
    (255, 0,   255),
]

# Enable/disable analysis
ENABLE_ANALYSIS = True
PRINT_STATS_INTERVAL = 30  # Print stats every N frames

# Load ONNX model
print("Loading ONNX model...")
session = ort.InferenceSession("violence_yolo.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("✓ Model loaded")

# Open video
print("Opening video...")
cap = cv2.VideoCapture("demovid/vid5.avi")  # or path to video

if not cap.isOpened():
    print("❌ Error: Cannot open video")
    exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✓ Video opened: {video_fps:.1f} FPS, {total_frames} frames")

# Initialize
frame_id   = 0
tracks    = []
raw_boxes = []
raw_confs = []

# Initialize analyzer
if ENABLE_ANALYSIS:
    analyzer = QuickAnalyzer()
    print("\n" + "="*70)
    print("STARTING WITH ANALYSIS ENABLED")
    print("="*70)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()
    H, W = frame.shape[:2]
    is_detect_frame = (frame_id % DETECT_INTERVAL == 0)

    inference_time = 0
    tracking_time = 0

    # ===== BETWEEN-FRAME TRACKING =====
    if not is_detect_frame:
        tracking_start = time.time()
        
        for t in tracks:
            if t['tracker'] is None or not t['tracker_ok']:
                continue

            ok, new_xywh = t['tracker'].update(frame)
            if ok:
                new_box = xywh_to_xyxy(new_xywh)
                if (new_box[0] >= 0 and new_box[1] >= 0 and
                    new_box[2] <= W  and new_box[3] <= H  and
                    new_box[2] > new_box[0] and new_box[3] > new_box[1]):
                    t['box'] = alpha * t['box'] + (1 - alpha) * new_box
                    t['tracker_ok'] = True
                else:
                    t['tracker_ok'] = False
            else:
                t['tracker_ok'] = False
        
        tracking_time = time.time() - tracking_start

    # ===== YOLO DETECTION =====
    if is_detect_frame:
        # Resize to model size (320x320)
        img, scale, pad_x, pad_y = letterbox_image(frame, (320, 320))

        # Convert uint8 → float32 with proper shape
        img_input = img.astype(np.float32) / 255.0  # uint8 to float32
        img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
        img_input = np.expand_dims(img_input, 0)  # Add batch dimension

        # Inference
        inference_start = time.time()
        outputs = session.run(None, {input_name: img_input}) #ONNX Runtime inference
        inference_time = time.time() - inference_start

        detections = outputs[0]     # (1, 5, 6)
        feature    = outputs[-1]     # (1, 512)

        detections = np.squeeze(detections, axis=0)
        feature    = np.squeeze(feature, axis=0)

        boxes = rescale_boxes(detections, scale, pad_x, pad_y, (H, W))  # rescale to original frame size
        boxes, areas = merge_overlapping_boxes(boxes)  # merge overlapping detections into one (optional, can help reduce noise)

        raw_boxes = []
        raw_confs = []

        if len(boxes) > 0:

            confs = boxes[:, 4]  # Extract confidence scores from the last column
            boxes = boxes[:, :4]  # Extract box coordinates
            #areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])

            if tracks:
                prev     = np.array([t['box'] for t in tracks])
                dist_mat = l1_center_dist_matrix(prev, boxes) / (W + H)
                best_stick = np.clip(1.0 - k * dist_mat.min(axis=0), 0.0, 1.0)
            else:
                best_stick = np.zeros(len(boxes))

            scores  = confs * areas * (1.0 + STICK_WEIGHT * best_stick)
            top_idx = np.argsort(scores)[::-1][:MAX_TRACKS]
            raw_boxes = boxes[top_idx]
            raw_confs = confs[top_idx]

        # ===== MATCH YOLO → TRACKS =====
        if len(raw_boxes) > 0:
            det_boxes = np.array(raw_boxes)
            det_confs = np.array(raw_confs)

            if not tracks:
                for i in range(len(det_boxes)):
                    tr = make_tracker()
                    tr.init(frame, xyxy_to_xywh(det_boxes[i]))
                    tracks.append({
                        'box':        det_boxes[i],
                        'conf':       det_confs[i],
                        'show':       False,
                        'tracker':    tr,
                        'tracker_ok': True,
                    })
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
                    tr = make_tracker()
                    tr.init(frame, xyxy_to_xywh(det_boxes[di]))
                    tracks[ti]['box']        = alpha * tracks[ti]['box'] + (1-alpha) * det_boxes[di]
                    tracks[ti]['conf']       = alpha * tracks[ti]['conf'] + (1-alpha) * det_confs[di]
                    tracks[ti]['tracker']    = tr
                    tracks[ti]['tracker_ok'] = True
                    matched_trk.add(ti)
                    matched_det.add(di)
                    if len(matched_trk) == min(len(tracks), len(det_boxes)):
                        break

                for di in range(len(det_boxes)):
                    if di not in matched_det and len(tracks) < MAX_TRACKS:
                        tr = make_tracker()
                        tr.init(frame, xyxy_to_xywh(det_boxes[di]))
                        tracks.append({
                            'box':        det_boxes[di],
                            'conf':       det_confs[di],
                            'show':       False,
                            'tracker':    tr,
                            'tracker_ok': True,
                        })

                for ti in range(len(tracks)):
                    if ti not in matched_trk:
                        tracks[ti]['conf']       *= 0.5
                        tracks[ti]['tracker_ok']  = False

        else:
            for t in tracks:
                t['conf']       *= 0.5
                t['tracker_ok']  = False

        tracks = [t for t in tracks if t['conf'] >= CONF_OFF]
        tracks = merge_overlapping_tracks(tracks) 

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

        status = "OK" if t['tracker_ok'] else "HOLD"
        cv2.putText(frame, f"#{i} {t['conf']:.2f} [{status}]", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ===== ANALYSIS =====
    frame_time = time.time() - frame_start
    
    if ENABLE_ANALYSIS:
        analyzer.record_frame(frame_time, inference_time, tracking_time)
        
        # Print stats periodically
        if (frame_id + 1) % PRINT_STATS_INTERVAL == 0:
            fps = analyzer.get_fps()
            memory = analyzer.get_memory_mb()
            times = analyzer.get_avg_times()
            
            print(f"\nFrame {frame_id+1:5d} | "
                  f"FPS: {fps:6.2f} | "
                  f"Frame: {times['frame']:6.2f}ms | "
                  f"Inference: {times['inference']:6.2f}ms | "
                  f"Memory: {memory:6.0f}MB | "
                  f"Tracks: {len(tracks)}")
    
    # Display
    if ENABLE_ANALYSIS:
        # Add analysis overlay
        cv2.putText(frame, f"FPS: {analyzer.get_fps():.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Memory: {analyzer.get_memory_mb():.0f}MB", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# Final summary
if ENABLE_ANALYSIS:
    print("\n" + "="*70)
    print("FINAL ANALYSIS SUMMARY")
    print("="*70)
    fps = analyzer.get_fps()
    memory = analyzer.get_memory_mb()
    times = analyzer.get_avg_times()
    elapsed = time.time() - analyzer.start_time
    
    print(f"Frames processed: {analyzer.frame_count}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    print(f"Avg frame time: {times['frame']:.2f} ms")
    print(f"Avg inference time: {times['inference']:.2f} ms")
    print(f"Avg tracking time: {times['tracking']:.2f} ms")
    print(f"Peak memory: {memory:.0f} MB")
    print("="*70)
