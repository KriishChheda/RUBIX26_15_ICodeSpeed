# AI Proctoring System

A modular, extensible AI-powered proctoring system built with Python and DeepFace. The system uses computer vision to monitor exam participants in real-time with support for face detection, face verification, and easy integration of custom detection modules.

## Features

- **Face Detection & Recognition**: Uses DeepFace library with multiple backend options (OpenCV, MTCNN, RetinaFace, etc.)
- **Face Verification**: Match detected faces against known participant photos
- **Modular Architecture**: Easy to add new detector modules (phone detection, object detection, etc.)
- **Performance Optimized**: Configurable frame skipping (process every Nth frame) for efficient CPU usage
- **Real-time Monitoring**: Live video feed with annotations and alerts
- **Comprehensive Reporting**: Detailed session reports with detection statistics

## Architecture

```
HD_ML_stuff/
├── modules/
│   ├── __init__.py
│   ├── base_detector.py        # Abstract base class for all detectors
│   ├── camera_input.py          # Camera capture handling
│   ├── camera_pipeline.py       # Base pipeline class
│   ├── proctor_pipeline.py      # Main proctoring pipeline (inherits CameraPipeline)
│   ├── face_detector.py         # DeepFace-based face detection & verification
│   ├── phone_detector.py        # Example: Phone detection template
│   ├── display.py               # Display window management
│   └── config.py                # System configuration
├── cv_models/                   # Directory for CV models
├── data/
│   └── participants/            # Participant face images for verification
│       ├── participant_1/
│       │   ├── photo_1.jpg
│       │   └── photo_2.jpg
│       └── participant_2/
├── main.py                      # Original face detection demo
├── proctor_main.py              # Proctoring system main application
├── requirements.txt
└── README.md
```

## Installation

1. **Clone or navigate to the repository**
```bash
cd d:\HD\Code\RUBIX26_15_ICodeSpeed\HD_ML_stuff
```

2. **Create virtual environment** (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

On first run, DeepFace will automatically download the required models.

## Quick Start

### Basic Proctoring Session

```bash
python proctor_main.py
```

This will:
- Initialize the camera
- Load face detection model
- Start monitoring with real-time display
- Process every 3rd frame (configurable)
- Generate a report on exit

### With Face Verification

To enable face verification against known participants:

1. Add participant photos to `data/participants/`:
```
data/participants/
├── john_doe/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── jane_smith/
    ├── photo1.jpg
    └── photo2.jpg
```

2. Enable verification in code:
```python
Config.FACE_VERIFICATION_ENABLED = True
```

3. Run the proctoring system

## Configuration

Edit [modules/config.py](modules/config.py) or configure in your main script:

```python
from modules import Config

# Camera settings
Config.CAMERA_ID = 0
Config.CAMERA_WIDTH = 1080
Config.CAMERA_HEIGHT = 720
Config.CAMERA_FPS = 30

# Performance
Config.FRAME_SKIP = 2  # Process every 3rd frame

# Face detection
Config.FACE_DETECTOR_BACKEND = "opencv"  # Fast
# Options: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe'

Config.FACE_MODEL_NAME = "Facenet"  # Accurate
# Options: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace'

Config.FACE_VERIFICATION_THRESHOLD = 0.4  # Lower = stricter
```

## Adding Custom Detectors

The system is designed for easy extension. Here's how to add a new detector:

### 1. Create Detector Class

```python
from modules.base_detector import BaseDetector

class MyCustomDetector(BaseDetector):
    def __init__(self, name="MyDetector", enabled=True):
        super().__init__(name, enabled)
        self.model = None
    
    def load_model(self):
        # Load your model here
        self.model = load_my_model()
        self.initialized = True
        return True
    
    def process_frame(self, frame, draw=True):
        if not self.enabled:
            return frame, {"enabled": False}
        
        # Your detection logic
        detections = self.model.detect(frame)
        
        # Annotate frame
        if draw:
            frame = self.draw_annotations(frame, detections)
        
        # Return results
        results = {
            "detector": self.name,
            "detections": detections,
            "count": len(detections)
        }
        
        return frame, results
```

### 2. Register with Proctor

```python
from modules import ProctorPipeline
from my_detector import MyCustomDetector

# Create pipeline
proctor = ProctorPipeline(frame_skip=2)

# Add detectors
face_detector = FaceDetector()
custom_detector = MyCustomDetector()

proctor.register_detector(face_detector)
proctor.register_detector(custom_detector)

# Run
proctor.run()
```

### Example: Phone Detector

See [modules/phone_detector.py](modules/phone_detector.py) for a complete template showing how to implement a phone detection module.

## Frame Processing Strategy

The system processes every Nth frame to optimize performance:

- **FRAME_SKIP = 0**: Process all frames (~30 FPS)
- **FRAME_SKIP = 1**: Process every 2nd frame (~15 FPS)  
- **FRAME_SKIP = 2**: Process every 3rd frame (~10 FPS) ✓ **Recommended**

At 30 FPS camera rate:
- Processing every 3rd frame = ~10 FPS processing rate
- Frame interval = 100ms (sufficient for proctoring)
- Reduces CPU usage significantly
- Minimal impact on detection accuracy

## DeepFace Backends Comparison

| Backend | Speed | Accuracy | Notes |
|---------|-------|----------|-------|
| opencv | ★★★★★ | ★★★ | Fast, good for real-time |
| ssd | ★★★★ | ★★★★ | Balanced |
| mtcnn | ★★★ | ★★★★★ | Very accurate, slower |
| retinaface | ★★ | ★★★★★ | Best accuracy, slowest |
| mediapipe | ★★★★★ | ★★★★ | Fast and accurate |

## API Reference

### ProctorPipeline

Main proctoring class that inherits from `CameraPipeline`.

```python
proctor = ProctorPipeline(config=Config, frame_skip=2)

# Detector management
proctor.register_detector(detector)
proctor.unregister_detector(detector_name)
proctor.get_detector(detector_name)
proctor.list_detectors()

# Control
proctor.enable_detector(detector_name)
proctor.disable_detector(detector_name)

# Run
proctor.run()

# Get report
report = proctor.get_proctoring_report()
```

### FaceDetector

DeepFace-based face detection and verification.

```python
face_detector = FaceDetector(
    name="FaceDetector",
    enabled=True,
    detector_backend='opencv',
    model_name='Facenet',
    verification_threshold=0.4,
    participant_data_path='data/participants'
)

# Detect faces
face_objs = face_detector.detect_faces(frame)

# Verify against known participants
result = face_detector.verify_face(frame, participant_id="john_doe")

# Process frame (detect + annotate)
annotated_frame, results = face_detector.process_frame(frame)
```

### BaseDetector

Abstract base class for all detectors.

```python
class MyDetector(BaseDetector):
    def load_model(self):
        # Initialize model
        pass
    
    def process_frame(self, frame):
        # Detect and return (frame, results)
        pass
```

## Proctoring Report

On session end, a comprehensive report is generated:

```
PROCTORING SESSION REPORT
==================================================================
Total Frames Captured: 3000
Total Frames Processed: 1000
Processing Ratio: 33.33%
Total Alerts: 15

Alert Summary:
  - multiple_faces: 5
  - no_face: 10
==================================================================
```

## Controls

- **ESC** or **Q**: Exit proctoring session
- Reports are automatically generated on exit

## Troubleshooting

### DeepFace model download fails
- Check internet connection
- Models are downloaded to `~/.deepface/weights/`
- Manual download: Visit DeepFace GitHub

### Camera not detected
- Check `Config.CAMERA_ID` (try 0, 1, 2...)
- Verify camera permissions
- Test with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Low FPS
- Increase `Config.FRAME_SKIP`
- Use faster backend: `opencv` or `mediapipe`
- Reduce camera resolution

### Face verification not working
- Ensure participant photos are clear and well-lit
- Use multiple photos per participant (3-5 recommended)
- Adjust `FACE_VERIFICATION_THRESHOLD` (lower = stricter)

## Future Enhancements

- [ ] Phone detection model integration
- [ ] Multiple person tracking
- [ ] Gaze detection (looking away from screen)
- [ ] Audio monitoring
- [ ] Cloud-based result storage
- [ ] Real-time alerts/notifications
- [ ] Web dashboard

## Dependencies

- `opencv-python` >= 4.13.0.90 - Computer vision
- `deepface` >= 0.0.79 - Face recognition
- `tensorflow` >= 2.12.0 - DeepFace backend
- `numpy` >= 1.24.0 - Array operations
- `Pillow` >= 9.5.0 - Image processing

## License

This project is for educational and research purposes.

## Contributing

To add new detector modules:
1. Inherit from `BaseDetector`
2. Implement `load_model()` and `process_frame()`
3. Place models in `cv_models/` directory
4. Register with `ProctorPipeline`

## Author

HD - 2026

---

**Note**: This system is designed for educational purposes. Always ensure compliance with privacy laws and regulations when using proctoring systems.
