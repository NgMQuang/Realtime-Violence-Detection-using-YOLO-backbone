"""
PIPELINE ANALYSIS TOOL
Tracks FPS, runtime, resources, and bottlenecks in your violence detection system
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import psutil
import os
from collections import deque
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, '.')
from run.utilities import *


class PipelineAnalyzer:
    """
    Analyzes performance of the violence detection pipeline.
    Tracks FPS, latency, memory usage, and identifies bottlenecks.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Args:
            max_history: Number of frames to keep history for (for averaging)
        """
        self.max_history = max_history
        
        # Timing data
        self.frame_times = deque(maxlen=max_history)
        self.letterbox_times = deque(maxlen=max_history)
        self.preprocess_times = deque(maxlen=max_history)
        self.inference_times = deque(maxlen=max_history)
        self.postprocess_times = deque(maxlen=max_history)
        self.tracking_times = deque(maxlen=max_history)
        self.draw_times = deque(maxlen=max_history)
        
        # Detection counts
        self.detection_counts = deque(maxlen=max_history)
        self.track_counts = deque(maxlen=max_history)
        
        # Memory data
        self.memory_usage = deque(maxlen=max_history)
        
        # Process info
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        
        # Frame counter
        self.frame_count = 0
        
    def get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident set size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual memory size
            'percent': self.process.memory_percent(),
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent(interval=0.01)
    
    def record_frame_time(self, total_time: float):
        """Record total frame time"""
        self.frame_times.append(total_time)
        self.frame_count += 1
    
    def record_letterbox_time(self, elapsed: float):
        """Record letterbox operation time"""
        self.letterbox_times.append(elapsed)
    
    def record_preprocess_time(self, elapsed: float):
        """Record preprocessing time"""
        self.preprocess_times.append(elapsed)
    
    def record_inference_time(self, elapsed: float):
        """Record ONNX inference time"""
        self.inference_times.append(elapsed)
    
    def record_postprocess_time(self, elapsed: float):
        """Record postprocessing time"""
        self.postprocess_times.append(elapsed)
    
    def record_tracking_time(self, elapsed: float):
        """Record tracking time"""
        self.tracking_times.append(elapsed)
    
    def record_draw_time(self, elapsed: float):
        """Record drawing time"""
        self.draw_times.append(elapsed)
    
    def record_detections(self, count: int):
        """Record number of detections"""
        self.detection_counts.append(count)
    
    def record_tracks(self, count: int):
        """Record number of active tracks"""
        self.track_counts.append(count)
    
    def record_memory(self):
        """Record current memory usage"""
        mem = self.get_memory_usage()
        self.memory_usage.append(mem)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        if not self.frame_times:
            return {}
        
        frame_times_list = list(self.frame_times)
        avg_frame_time = np.mean(frame_times_list)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Calculate FPS values safely
        fps_values = [1.0 / t for t in frame_times_list if t > 0]
        fps_min = min(fps_values) if fps_values else 0
        fps_max = max(fps_values) if fps_values else 0
        fps_std = np.std(fps_values) if fps_values else 0
        
        stats = {
            'frame_count': self.frame_count,
            'elapsed_time': time.time() - self.start_time,
            
            # FPS
            'fps': fps,
            'fps_min': fps_min,
            'fps_max': fps_max,
            'fps_std': fps_std,
            
            # Frame timing (ms)
            'frame_time_avg': avg_frame_time * 1000,
            'frame_time_min': min(frame_times_list) * 1000,
            'frame_time_max': max(frame_times_list) * 1000,
            
            # Component timing (ms)
            'letterbox_time_avg': np.mean(list(self.letterbox_times)) * 1000 if self.letterbox_times else 0,
            'preprocess_time_avg': np.mean(list(self.preprocess_times)) * 1000 if self.preprocess_times else 0,
            'inference_time_avg': np.mean(list(self.inference_times)) * 1000 if self.inference_times else 0,
            'postprocess_time_avg': np.mean(list(self.postprocess_times)) * 1000 if self.postprocess_times else 0,
            'tracking_time_avg': np.mean(list(self.tracking_times)) * 1000 if self.tracking_times else 0,
            'draw_time_avg': np.mean(list(self.draw_times)) * 1000 if self.draw_times else 0,
            
            # Memory
            'memory_mb': self.memory_usage[-1]['rss_mb'] if self.memory_usage else 0,
            'memory_percent': self.memory_usage[-1]['percent'] if self.memory_usage else 0,
            'memory_max_mb': max([m['rss_mb'] for m in self.memory_usage]) if self.memory_usage else 0,
            
            # Detections and tracks
            'avg_detections': np.mean(list(self.detection_counts)) if self.detection_counts else 0,
            'avg_tracks': np.mean(list(self.track_counts)) if self.track_counts else 0,
        }
        
        return stats
    
    def print_summary(self):
        """Print summary statistics"""
        stats = self.get_stats()
        
        if not stats:
            print("No data collected yet")
            return
        
        print("\n" + "="*70)
        print("PIPELINE ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Frames processed: {stats['frame_count']}")
        print(f"  Elapsed time: {stats['elapsed_time']:.2f} seconds")
        print(f"  Average FPS: {stats['fps']:.2f}")
        print(f"  FPS range: {stats['fps_min']:.2f} - {stats['fps_max']:.2f}")
        print(f"  FPS std dev: {stats['fps_std']:.2f}")
        
        print(f"\n⏱️  FRAME TIMING:")
        print(f"  Avg frame time: {stats['frame_time_avg']:.2f} ms")
        print(f"  Min frame time: {stats['frame_time_min']:.2f} ms")
        print(f"  Max frame time: {stats['frame_time_max']:.2f} ms")
        print(f"  Budget for 30 FPS: 33.33 ms")
        
        print(f"\n🔍 COMPONENT TIMING (avg ms per frame):")
        components = [
            ('Letterbox', stats['letterbox_time_avg']),
            ('Preprocess', stats['preprocess_time_avg']),
            ('Inference', stats['inference_time_avg']),
            ('Postprocess', stats['postprocess_time_avg']),
            ('Tracking', stats['tracking_time_avg']),
            ('Drawing', stats['draw_time_avg']),
        ]
        
        total_component_time = sum(t for _, t in components)
        
        for name, time_ms in sorted(components, key=lambda x: x[1], reverse=True):
            percent = (time_ms / stats['frame_time_avg'] * 100) if stats['frame_time_avg'] > 0 else 0
            bar = '█' * int(percent / 2)
            print(f"  {name:12s}: {time_ms:6.2f} ms ({percent:5.1f}%) {bar}")
        
        print(f"\n💾 MEMORY USAGE:")
        print(f"  Current: {stats['memory_mb']:.1f} MB")
        print(f"  Peak: {stats['memory_max_mb']:.1f} MB")
        print(f"  CPU usage: {stats['memory_percent']:.1f}%")
        
        print(f"\n📍 DETECTION & TRACKING:")
        print(f"  Avg detections per frame: {stats['avg_detections']:.2f}")
        print(f"  Avg active tracks: {stats['avg_tracks']:.2f}")
        
        print("\n" + "="*70)
    
    def print_bottleneck_analysis(self):
        """Identify and report bottlenecks"""
        stats = self.get_stats()
        
        if not stats:
            return
        
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS")
        print("="*70)
        
        # Identify slowest components
        components = [
            ('Inference', stats['inference_time_avg']),
            ('Tracking', stats['tracking_time_avg']),
            ('Postprocess', stats['postprocess_time_avg']),
            ('Drawing', stats['draw_time_avg']),
            ('Letterbox', stats['letterbox_time_avg']),
            ('Preprocess', stats['preprocess_time_avg']),
        ]
        
        components_sorted = sorted(components, key=lambda x: x[1], reverse=True)
        
        print("\nSlowest components (most impact on frame time):")
        for i, (name, time_ms) in enumerate(components_sorted[:3], 1):
            percent = (time_ms / stats['frame_time_avg'] * 100) if stats['frame_time_avg'] > 0 else 0
            print(f"  {i}. {name:12s}: {time_ms:6.2f} ms ({percent:5.1f}%)")
        
        # FPS analysis
        print(f"\nFPS Performance:")
        target_fps = 30
        avg_fps = stats['fps']
        
        if avg_fps >= target_fps:
            print(f"  ✓ EXCEEDS TARGET: {avg_fps:.1f} FPS (target: {target_fps})")
        else:
            deficit = target_fps - avg_fps
            print(f"  ⚠️  BELOW TARGET: {avg_fps:.1f} FPS (target: {target_fps}, deficit: {deficit:.1f})")
            needed_speedup = (stats['frame_time_avg'] / (1.0 / target_fps)) * 100
            print(f"  Need {needed_speedup:.0f}% speedup to reach target")
        
        # Memory analysis
        print(f"\nMemory Usage:")
        peak_memory = stats['memory_max_mb']
        avg_memory = stats['memory_mb']
        
        if peak_memory < 1000:
            print(f"  ✓ GOOD: {peak_memory:.1f} MB peak")
        elif peak_memory < 2000:
            print(f"  ⚠️  MODERATE: {peak_memory:.1f} MB peak")
        else:
            print(f"  ⚠️  HIGH: {peak_memory:.1f} MB peak")
        
        print("\n" + "="*70)


def run_analysis(
    video_path: str,
    model_path: str,
    max_frames: int = 300,
    analyze_interval: int = 30
):
    """
    Run complete pipeline analysis.
    
    Args:
        video_path: Path to input video
        model_path: Path to ONNX model
        max_frames: Maximum frames to process
        analyze_interval: Print stats every N frames
    """
    
    print("="*70)
    print("STARTING PIPELINE ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = PipelineAnalyzer()
    
    # Load model
    print("\n📦 Loading ONNX model...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"   ✓ Model loaded: {model_path}")
    
    # Open video
    print(f"\n📹 Opening video...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"   ❌ Error: Cannot open video {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   ✓ Video opened: {video_fps:.1f} FPS, {total_frames} frames")
    
    # Config (same as your run.py)
    FPS_VIDEO = int(video_fps) if video_fps > 0 else 30
    TOTAL_TIME_DETECT = 2.5
    FRAME_PER_DETECT = 8
    DETECT_INTERVAL = round(FPS_VIDEO * TOTAL_TIME_DETECT / FRAME_PER_DETECT)
    
    CONF_ON = 0.25
    CONF_OFF = 0.1
    alpha = 0.8
    MAX_TRACKS   = 5
    
    print(f"   Config: Detect every {DETECT_INTERVAL} frames")
    
    # Initialize tracking
    frame_id = 0
    tracks = []
    
    print("\n▶️  Starting processing...\n")
    
    # Process frames
    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        H, W = frame.shape[:2]
        is_detect_frame = (frame_id % DETECT_INTERVAL == 0)
        
        # ===== BETWEEN-FRAME TRACKING =====
        tracking_start = time.time()
        
        if not is_detect_frame:
            for t in tracks:
                if t['tracker'] is not None and t['tracker_ok']:
                    ok, new_xywh = t['tracker'].update(frame)
                    if ok:
                        new_box = xywh_to_xyxy(new_xywh)
                        if (new_box[0] >= 0 and new_box[1] >= 0 and
                            new_box[2] <= W and new_box[3] <= H and
                            new_box[2] > new_box[0] and new_box[3] > new_box[1]):
                            t['box'] = alpha * t['box'] + (1 - alpha) * new_box
                            t['tracker_ok'] = True
                        else:
                            t['tracker_ok'] = False
                    else:
                        t['tracker_ok'] = False
        
        tracking_time = time.time() - tracking_start
        analyzer.record_tracking_time(tracking_time)
        
        # ===== YOLO DETECTION =====
        if is_detect_frame:
            # Letterbox
            letterbox_start = time.time()
            img, scale, pad_x, pad_y = letterbox_image(frame, (320, 320))
            analyzer.record_letterbox_time(time.time() - letterbox_start)
            
            # Preprocess
            preprocess_start = time.time()
            img_input = img.astype(np.float32) / 255.0
            img_input = np.transpose(img_input, (2, 0, 1))
            img_input = np.expand_dims(img_input, 0)
            analyzer.record_preprocess_time(time.time() - preprocess_start)
            
            # Inference
            inference_start = time.time()
            outputs = session.run(None, {input_name: img_input})
            analyzer.record_inference_time(time.time() - inference_start)
            
            # Postprocess
            postprocess_start = time.time()
            detections = outputs[0]
            detections = np.squeeze(detections, axis=0)
            
            boxes = rescale_boxes(detections, scale, pad_x, pad_y, (H, W))
            boxes, areas = merge_overlapping_boxes(boxes)  # merge overlapping detections into one (optional, can help reduce noise)
            
            raw_boxes = []
            raw_confs = []
            
            if len(boxes) > 0:
                confs = boxes[:, 4]
                boxes_coords = boxes[:, :4]
                
                if tracks:
                    prev = np.array([t['box'] for t in tracks])
                    dist_mat = l1_center_dist_matrix(prev, boxes_coords) / (W + H)
                    best_stick = np.clip(1.0 - 5.0 * dist_mat.min(axis=0), 0.0, 1.0)
                else:
                    best_stick = np.zeros(len(boxes_coords))
                
                scores = confs * areas * (1.0 + 0.7 * best_stick)
                top_idx = np.argsort(scores)[::-1][:5]
                raw_boxes = boxes_coords[top_idx]
                raw_confs = confs[top_idx]
            
            analyzer.record_detections(len(raw_boxes))
            analyzer.record_postprocess_time(time.time() - postprocess_start)
            
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
        
        # ===== HYSTERESIS & DRAW =====
        draw_start = time.time()
        
        for t in tracks:
            if t['conf'] >= CONF_ON:
                t['show'] = True
            elif t['conf'] < CONF_OFF:
                t['show'] = False
        
        for i, t in enumerate(tracks):
            if not t['show']:
                continue
            color = (0, 255, 0)
            x1, y1, x2, y2 = map(int, t['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            status = "OK" if t['tracker_ok'] else "HOLD"
            cv2.putText(frame, f"#{i} {t['conf']:.2f} [{status}]", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        analyzer.record_draw_time(time.time() - draw_start)
        analyzer.record_tracks(len(tracks))
        
        # Record frame time and memory
        frame_time = time.time() - frame_start
        analyzer.record_frame_time(frame_time)
        analyzer.record_memory()
        
        # Display frame (optional, slows down analysis)
        cv2.imshow("Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print progress
        if (frame_id + 1) % analyze_interval == 0:
            stats = analyzer.get_stats()
            print(f"Frame {frame_id+1}/{max_frames} | "
                  f"FPS: {stats['fps']:.1f} | "
                  f"Detections: {stats['avg_detections']:.1f} | "
                  f"Memory: {stats['memory_mb']:.0f} MB")
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final analysis
    print("\n")
    analyzer.print_summary()
    analyzer.print_bottleneck_analysis()


if __name__ == "__main__":
    # Run analysis
    run_analysis(
        video_path="demovid/vid5.avi",
        model_path="violence_yolo.onnx",
        max_frames=300,
        analyze_interval=30
    )