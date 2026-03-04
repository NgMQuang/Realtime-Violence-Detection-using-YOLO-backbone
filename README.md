Note: Claude write the template, I'm updating
# Violence Detection System with YOLO26 + Temporal Classifier

Real-time violence detection using multi-object tracking and temporal classification. Detects and localizes violent behavior in video with bounding boxes and per-frame violence probability scores.

## 🎯 Features

- **Spatial Detection**: YOLO26 for real-time object detection and localization
- **Multi-Object Tracking**: MOSSE tracker with confidence-based hysteresis
- **Temporal Classification**: GAPConv1D classifier on 8-frame feature sequences
- **Robust Tracking**: Automatic tracker failure recovery and confidence decay
- **Real-time Performance**: Optimized for CPU and GPU inference
- **Adaptive Frame Sampling**: Configurable detection intervals for resource efficiency

## 📊 Performance

| Metric | Value |
|--------|-------|
| mAP0.5 (Custom Dataset) | **0.75** |
| Accuracy (Custom Dataset) | **82.63%** |
| Dataset Source | RWF2000 |
| Input Resolution | 320×320 |

**Validation Results**:

|Accuracy | 0.8263|
|---------|-------|
|Precision| 0.8182|
|Recall   | 0.8424|
|F1 Score | 0.8301|


|Class     |Images|  Instances | Box(P    -      R    -  mAP50  -    mAP50-95)|
|----------|------|------------|----------------------------------------------|
|all       |3000  |    2865    |  0.712   -   0.704   -  0.754  -       0.425 |

## 🏗️ Architecture

```
Video Frame
    ↓
[YOLO Detection] → Bounding boxes + Confidence
    ↓
[MOSSE Tracker] → Track objects across frames
    ↓
[Feature Extraction] → 512-dim feature vectors
    ↓
[GAPConv1D Classifier] → Violence probability (8-frame window)
    ↓
Output: Labeled frame with boxes + violence score
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- ONNX Runtime
- OpenCV
- NumPy

### Setup

```bash
# Clone repository
git clone https://github.com/QuangNgM/violence-detection.git
cd violence-detection

# Install dependencies
pip install -r requirements.txt

# Install Git LFS (for model files)
git lfs install
git lfs pull
```

## 🚀 Usage

### Basic Usage

```bash
python run.py
```

By default, it processes `demovid/vid1.avi`. Edit the video path in `run.py`:

```python
cap = cv2.VideoCapture("path/to/your/video.avi")
```

### Configuration

Edit parameters in `run.py`:

```python
FPS_VIDEO = 30                    # Video frame rate
TOTAL_TIME_DETECT = 2.5          # Detection window (seconds)
FRAME_PER_DETECT = 8             # Frames per classifier input
DETECT_INTERVAL = 10             # Frames between detections

TRACKER = "MOSSE"                # Tracker type
MAX_TRACKS = 5                   # Max simultaneous tracks
CONF_ON = 0.25                   # Show track threshold
CONF_OFF = 0.1                   # Hide track threshold
STICK_WEIGHT = 0.7               # Stickiness in scoring
alpha = 0.8                      # EMA smoothing factor
TRACKER_FAILURE_DECAY = 0.5      # Confidence decay on failure
```

## 📁 Project Structure

```
violence-detection/
├── demovid                     # Videos for demo (removed on github)
├── run.py                      # Main inference script
├── utilities.py                # Helper functions (tracking, box operations)
├── pipeline_analyzer.py        # Model analysis/debugging
├── run_with_analysis.py        # Inference with detailed logging
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # License
├── violence_yolo.onnx      # YOLO26 detection model
├── gapconv1d.onnx          # Temporal classification model
└── gapconv1d.onnx.data     # Model weights (LFS tracked)

```

## 🔍 How It Works

### 1. Detection Phase (Every N frames)
- YOLO detects suspicious areas
- Returns bounding boxes with confidence scores
- Extracts 512-dimensional feature vectors

### 2. Tracking Phase (Between detections)
- Tracker(MOSSE) updates box positions frame-to-frame
- Confidence scores decay if tracker fails
- Boxes with low confidence are removed

### 3. Classification Phase (Every N frames)
- Collects last 8 feature vectors
- Passes to GAPConv1D temporal classifier
- Outputs violence probability (0.0 - 1.0)
- Alerts if probability > 0.8

### 4. Hysteresis & Display
- Tracks shown/hidden based on confidence thresholds
- Color-coded bounding boxes with track IDs
- Violence probability displayed on frame

## 📊 Model Details

### YOLO26 (violence_yolo.onnx)
- **Input**: 320×320 RGB images (normalized 0-1)
- **Output**: 
  - Detections: (5, 6) - up to 5 boxes with [x1, y1, x2, y2, conf, class]
  - Features: (512,) - feature vector for temporal analysis
- **Inference Time**: ~50-100ms (CPU)

### GAPConv1D (gapconv1d.onnx)
- **Input**: (8, 512) - 8 consecutive feature vectors
- **Output**: (1,) - logit for binary classification
- **Processing**: Global Average Pooling + Conv1D
- **Inference Time**: ~5-10ms (CPU)

## 🎮 Output

The script displays:
- **Bounding boxes** around detected people
- **Track IDs** and confidence scores
- **Tracker status** ("OK" or "HOLD")
- **Violence probability** when classified
- **Alert** when violence confidence > 0.8

### Keyboard Controls
- `Q` - Quit

## ⚙️ Advanced Usage

### Use Custom Video Input

```python
# Webcam
cap = cv2.VideoCapture(0)

# IP camera
cap = cv2.VideoCapture("rtsp://your-camera-ip/stream")

# Video file
cap = cv2.VideoCapture("path/to/video.mp4")
```

### Adjust Violence Threshold

```python
if violence_prob > 0.8:  # Change threshold here
    logger.warning(f"HIGH VIOLENCE DETECTED!")
```

### Enable Detailed Analysis

```bash
python run_with_analysis.py
```

## 🐛 Troubleshooting

### Models not found
```bash
git lfs pull
# or manually download from releases
```

### Slow performance
- Increase `DETECT_INTERVAL` (skip more frames between detections)
- Reduce input resolution (change 320 to 240 in code)
- Disable tracker for CPU-only systems

### ONNX Runtime errors
```bash
pip install --upgrade onnxruntime
```

### No detections appearing
- Check video file is readable: `ffprobe video.avi`
- Verify model files exist: `ls -la models/`
- Run with `run_with_analysis.py` for debugging

## 🔬 Dataset

Models trained on custom dataset derived from **RWF2000** (Real World Fighting Dataset):
- Contains real-world violence/non-violence scenarios
- Custom labeling
- 0.75 mAP on detection task
- 82.63% accuracy on violence classification

## 📚 References

- YOLO26: https://github.com/ultralytics/ultralytics
- ONNX Runtime: https://onnxruntime.ai/
- RWF2000 Dataset: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

## 📄 License

MIT

## 🙏 Acknowledgments

- YOLO26 by Ultralytics
- RWF2000 dataset creators
- ONNX Runtime community

---

## Quick Start Checklist

- [ ] Clone repository
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Pull LFS models: `git lfs pull`
- [ ] Place your video in project folder
- [ ] Edit video path in `run.py`
- [ ] Run: `python run.py`
- [ ] Press Q to quit
