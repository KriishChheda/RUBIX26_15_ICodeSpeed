# YOLOv8 Face Detection Pipeline

A modular camera pipeline that captures video from a camera feed and performs real-time face/person detection using YOLOv8.

## Features

- **YOLOv8 Integration**: Real-time object detection using ultralytics YOLOv8 models
- **Modular Architecture**: Separate modules for camera input, display, detection, and configuration
- **Flexible Detection**: Detect any COCO dataset objects (persons, faces, etc.)
- **Real-time Performance**: Optimized for live camera feed processing
- **FPS Display**: Real-time FPS counter and detection count
- **Configurable**: Centralized configuration for all pipeline settings
- **Context Manager Support**: Automatic resource cleanup
- **Comprehensive Logging**: Track pipeline operations and errors

## Project Structure

```
camera_pipeline/
├── main.py                      # Main application entry point
├── modules/
│   ├── __init__.py             # Module exports
│   ├── pipeline.py             # Base CameraPipeline class
│   ├── camera_input.py         # Camera capture module
│   ├── display.py              # Display window module
│   ├── face_detector.py        # YOLOv8 detector module
│   └── config.py               # Configuration settings
├── example_custom_processing.py # Example custom pipelines
└── README.md                    # This file
```

## How It Works

### Architecture Overview

The pipeline follows a modular architecture with clear separation of concerns:

1. **Camera Input Module** (`camera_input.py`)
   - Captures frames from the camera using OpenCV
   - Manages camera lifecycle (start, read, stop)
   - Configurable resolution and FPS

2. **YOLOv8 Detector Module** (`face_detector.py`)
   - Loads YOLOv8 models using `ultralytics` library
   - Performs object detection on each frame
   - Filters detections by target classes (e.g., person class)
   - Draws bounding boxes with confidence scores

3. **Display Module** (`display.py`)
   - Creates and manages OpenCV display window
   - Renders processed frames with overlays
   - Handles user input (exit keys)

4. **Pipeline Module** (`pipeline.py`)
   - Orchestrates the entire processing pipeline
   - Manages frame loop and FPS calculation
   - Provides extensible `process_frame()` hook

5. **Main Application** (`main.py`)
   - Extends base pipeline with face detection logic
   - Initializes YOLOv8 model
   - Processes frames and displays results

### Detection Flow

```
Camera Feed → Capture Frame → YOLOv8 Detection → Draw Boxes → Display → Repeat
```

1. **Initialization**: Pipeline loads camera, creates display window, and initializes YOLOv8 model
2. **Frame Capture**: Camera continuously captures frames
3. **Detection**: YOLOv8 model processes frame and returns bounding boxes
4. **Visualization**: Bounding boxes and labels are drawn on the frame
5. **Display**: Processed frame is shown with FPS and detection count
6. **Loop**: Process repeats until user presses 'q' or ESC

### YOLOv8 Model Loading

The pipeline uses the standard ultralytics approach:

```python
from ultralytics import YOLO

# Load model (downloads automatically on first run)
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.

# Run inference
results = model(frame, conf=0.5)
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Virtual environment (recommended)

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/Mac
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `opencv-python` - Camera capture and image processing
- `numpy` - Array operations
- `ultralytics` - YOLOv8 model
- `torch` - PyTorch backend for YOLOv8
- `torchvision` - Computer vision utilities

### Step 3: Run the Pipeline

```bash
cd camera_pipeline
python main.py
```

**On first run**, YOLOv8 will automatically download the model file (~12MB for yolov8s).

### Step 4: Exit

Press `q` or `ESC` to stop the pipeline.

## Configuration Options

Edit [modules/config.py](modules/config.py) or override in your code:

### Camera Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_ID` | 0 | Camera device ID (0 for default webcam) |
| `CAMERA_WIDTH` | 1080 | Camera frame width in pixels |
| `CAMERA_HEIGHT` | 720 | Camera frame height in pixels |
| `CAMERA_FPS` | 30 | Camera frame rate |

### Display Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `WINDOW_NAME` | "Camera Pipeline" | Display window title |
| `FULLSCREEN` | False | Enable fullscreen mode |
| `SHOW_FPS` | True | Display FPS counter |

### Detection Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `FACE_MODEL_TYPE` | "yolov8" | Detector type ("yolov8" or "haar") |
| `FACE_MODEL_NAME` | "yolov8s.pt" | YOLOv8 model name |
| `FACE_CONFIDENCE_THRESHOLD` | 0.5 | Minimum detection confidence (0.0-1.0) |
| `FACE_TARGET_CLASSES` | [0] | COCO class IDs to detect ([0] = person) |
| `FACE_BOX_COLOR` | (0, 255, 0) | Bounding box color (B, G, R) |
| `FACE_BOX_THICKNESS` | 2 | Bounding box line thickness |
| `FACE_SHOW_CONFIDENCE` | True | Show confidence scores |

### Performance Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_FPS` | 60 | Maximum FPS to process |
| `FRAME_SKIP` | 0 | Number of frames to skip (0 = no skip) |
| `ENABLE_LOGGING` | True | Enable console logging |
| `LOG_LEVEL` | "INFO" | Logging level |

## Usage Examples

### Basic Usage

Run with default settings (person detection):

```bash
python main.py
```

Press `ESC` or `q` to exit.

### Custom Model Selection

Choose different YOLOv8 models for speed/accuracy tradeoff:

```python
from modules import Config

# Nano model (fastest, less accurate)
Config.FACE_MODEL_NAME = "yolov8n.pt"

# Small model (balanced)
Config.FACE_MODEL_NAME = "yolov8s.pt"

# Medium model (more accurate, slower)
Config.FACE_MODEL_NAME = "yolov8m.pt"

# Large model (most accurate, slowest)
Config.FACE_MODEL_NAME = "yolov8l.pt"
```

### Detect Multiple Object Types

Detect different objects using COCO class IDs:

```python
from modules import Config

# Detect persons only (default)
Config.FACE_TARGET_CLASSES = [0]

# Detect persons and cars
Config.FACE_TARGET_CLASSES = [0, 2]

# Detect all objects
Config.FACE_TARGET_CLASSES = None

# Common COCO class IDs:
# 0: person, 1: bicycle, 2: car, 3: motorcycle
# 16: dog, 17: cat, 39: bottle, 41: cup
```

### Adjust Detection Sensitivity

```python
from modules import Config

# Lower threshold = more detections (may include false positives)
Config.FACE_CONFIDENCE_THRESHOLD = 0.3

# Higher threshold = fewer detections (more accurate)
Config.FACE_CONFIDENCE_THRESHOLD = 0.7

# Customize camera settings
Config.CAMERA_WIDTH = 1280
Config.CAMERA_HEIGHT = 720
Config.SHOW_FPS = True
```

## Extending the Pipeline

### Custom Frame Processing

Create custom pipelines by extending the base class:

```python
from modules import CameraPipeline
import cv2

class CustomPipeline(CameraPipeline):
    def process_frame(self, frame):
        # Your custom processing here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Run custom pipeline
pipeline = CustomPipeline()
pipeline.run()
```

See [example_custom_processing.py](example_custom_processing.py) for more examples.

## Troubleshooting

### Camera Not Found

```
Error: Failed to open camera 0
```

**Solution**: Try different camera IDs or check camera permissions
```python
Config.CAMERA_ID = 1  # Try different IDs: 0, 1, 2...
```

### Model Download Issues

If automatic download fails, manually download from:
- YOLOv8 models: https://github.com/ultralytics/assets/releases

Place the `.pt` file in the camera_pipeline directory.

### Low FPS / Performance Issues

**Solutions:**
1. Use smaller model: `Config.FACE_MODEL_NAME = "yolov8n.pt"`
2. Reduce resolution: `Config.CAMERA_WIDTH = 640; Config.CAMERA_HEIGHT = 480`
3. Increase confidence threshold: `Config.FACE_CONFIDENCE_THRESHOLD = 0.6`
4. Enable frame skipping: `Config.FRAME_SKIP = 1`

### CUDA/GPU Support

For faster inference with NVIDIA GPU:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

YOLOv8 will automatically use GPU if available.

## Module Documentation

### CameraPipeline
Base pipeline class that orchestrates camera capture and display.

**Key Methods:**
- `initialize()`: Setup camera and display
- `run()`: Start pipeline loop
- `process_frame(frame)`: Override for custom processing
- `cleanup()`: Release all resources

### YOLOv8Detector
Object detection using YOLOv8 models.

**Key Methods:**
- `load_model()`: Initialize YOLOv8 model
- `detect_objects(frame, target_classes)`: Run detection
- `draw_boxes(frame, detections)`: Draw bounding boxes
- `process_frame(frame)`: Detect and draw in one step

### CameraCapture
Handles camera initialization, frame capture, and resource management.

**Key Methods:**
- `start()`: Initialize and start camera
- `read_frame()`: Read a single frame
- `stop()`: Release camera resources
- `get_properties()`: Get camera properties

### DisplayWindow
Manages display window and frame rendering.

**Key Methods:**
- `create_window()`: Create display window
- `show_frame(frame, fps)`: Display a frame
- `check_exit_key()`: Check for exit key press
- `destroy()`: Close window

## COCO Dataset Classes

The YOLOv8 models detect 80 object classes from the COCO dataset:

| Class ID | Object | Class ID | Object |
|----------|--------|----------|--------|
| 0 | person | 16 | dog |
| 1 | bicycle | 17 | cat |
| 2 | car | 24 | backpack |
| 3 | motorcycle | 26 | handbag |
| 5 | bus | 39 | bottle |
| 7 | truck | 41 | cup |
| 15 | bird | 67 | cell phone |

Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

## Performance Tips

1. **Model Selection**: 
   - `yolov8n` (~6 FPS on CPU, most real-time)
   - `yolov8s` (~4 FPS on CPU, balanced)
   - `yolov8m/l/x` (GPU recommended)

2. **Resolution**: Lower resolution = faster processing
   - 640x480: Best for CPU
   - 1280x720: Recommended for GPU
   - 1920x1080: High-end GPU only

3. **Confidence Threshold**: Higher = faster (fewer boxes to draw)

## Requirements

- Python 3.8+
- OpenCV 4.8.0+
- NumPy 1.24.0+
- Ultralytics 8.0.0+
- PyTorch 2.0.0+
- Webcam or camera device

## License

MIT License

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review module documentation
3. Consult ultralytics YOLOv8 docs: https://docs.ultralytics.com
