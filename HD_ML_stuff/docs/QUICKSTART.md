# Quick Start Guide - AI Proctoring System

## Installation (5 minutes)

### 1. Ensure Python Environment is Active
```powershell
# Your virtual environment should already be active
# If not:
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

**Note**: First run will download DeepFace models (~100-200MB). This is automatic and one-time only.

## Run the Proctoring System (1 minute)

### Basic Proctoring (No Verification)
```powershell
python proctor_main.py
```

**What it does:**
- ✅ Starts your camera
- ✅ Detects faces in real-time
- ✅ Shows alerts if multiple faces or no face detected
- ✅ Processes every 3rd frame (efficient)
- ✅ Generates report on exit

**Controls:**
- Press **Q** or **ESC** to quit

## With Face Verification (5 minutes setup)

### 1. Add Participant Photos

Create folder structure:
```
data/participants/john_doe/
```

Add 2-3 clear photos of the participant:
```powershell
# Example structure
data/
  participants/
    john_doe/
      photo1.jpg
      photo2.jpg
      photo3.jpg
```

**Photo Requirements:**
- Clear, well-lit frontal face
- No sunglasses or masks
- Similar lighting to exam environment
- 2-5 photos per participant (more = better)

### 2. Enable Verification

Edit [proctor_main.py](proctor_main.py) line 12:
```python
Config.FACE_VERIFICATION_ENABLED = True  # Change from False to True
```

### 3. Run
```powershell
python proctor_main.py
```

## Understanding the Display

### On-Screen Information

```
┌─────────────────────────────────────────┐
│  FPS: 29.3                              │  ← Camera FPS
│  FaceDetector: Verified (1 faces)       │  ← Green = Verified
│  PhoneDetector: Active                  │  ← Additional detectors
│                                         │
│  [Camera Feed with Boxes]              │
│                                         │
│  Processed: 1000/3000                   │  ← Processing stats
└─────────────────────────────────────────┘
```

### Box Colors
- **Green Box**: Face detected and verified ✅
- **Orange Box**: Face detected but not verified ⚠️
- **Red Box**: Alert condition (phone, suspicious object) 🚨

## Test Your Setup

### Quick Camera Test
```powershell
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"
```

### Module Import Test
```powershell
python -c "from modules import ProctorPipeline, FaceDetector; print('✓ Modules OK')"
```

### Full System Test
```powershell
python proctor_main.py
```
- Camera should open
- Your face should be detected with green box
- Press Q to quit
- Report should print to console

## Example Usage Patterns

### High Performance (Low CPU)
Edit [proctor_main.py](proctor_main.py):
```python
Config.FRAME_SKIP = 4  # Process every 5th frame
Config.FACE_DETECTOR_BACKEND = "opencv"  # Fastest
Config.CAMERA_WIDTH = 640
Config.CAMERA_HEIGHT = 480
```

### High Accuracy
```python
Config.FRAME_SKIP = 1  # Process every 2nd frame
Config.FACE_DETECTOR_BACKEND = "retinaface"  # Most accurate
Config.FACE_MODEL_NAME = "ArcFace"  # Best recognition
```

### Balanced (Recommended)
```python
Config.FRAME_SKIP = 2  # Process every 3rd frame ✓
Config.FACE_DETECTOR_BACKEND = "opencv"  # Fast ✓
Config.FACE_MODEL_NAME = "Facenet"  # Accurate ✓
```

## Common Issues & Solutions

### Issue: Camera not found
```
ERROR: Failed to start camera
```
**Solution:**
```python
# Try different camera IDs
Config.CAMERA_ID = 0  # Try 0, 1, 2...
```

### Issue: DeepFace models downloading slowly
```
Downloading: model.h5
```
**Solution:** 
- This is normal on first run
- Models download to `~/.deepface/weights/`
- Only happens once per model
- Total size: ~100-200MB

### Issue: Low FPS / Laggy
```
FPS: 5.2
```
**Solution:**
```python
Config.FRAME_SKIP = 4  # Increase frame skip
Config.CAMERA_WIDTH = 640  # Lower resolution
Config.FACE_DETECTOR_BACKEND = "opencv"  # Faster backend
```

### Issue: Face not detected
```
No face detected
```
**Solution:**
- Ensure good lighting
- Face camera directly
- Remove sunglasses/masks
- Try different backend:
```python
Config.FACE_DETECTOR_BACKEND = "mediapipe"  # or "mtcnn"
```

### Issue: Face not verified
```
Unverified (1 faces)
```
**Solution:**
- Add more participant photos (3-5 recommended)
- Ensure photos are clear and well-lit
- Adjust threshold:
```python
Config.FACE_VERIFICATION_THRESHOLD = 0.5  # Less strict (higher value)
```

## Customization

### Change Window Title
```python
Config.WINDOW_NAME = "My Exam Proctor"
```

### Disable Alerts
```python
Config.ENABLE_MULTI_FACE_ALERT = False
Config.ENABLE_NO_FACE_ALERT = False
```

### Change Camera Resolution
```python
Config.CAMERA_WIDTH = 1920
Config.CAMERA_HEIGHT = 1080
```

### Full Screen Mode
```python
Config.FULLSCREEN = True
```

## Understanding Frame Skip

```
Camera: 30 FPS (all frames)
    ↓
FRAME_SKIP = 2
    ↓
Process: Every 3rd frame
    ↓
Effective: ~10 FPS processing
    ↓
Result: 67% less CPU usage ✓
```

**Why it works for proctoring:**
- At 10 FPS, each frame is 100ms apart
- Human movements are much slower
- Cheating behaviors happen over seconds, not milliseconds
- Minimal detection accuracy impact
- Huge performance benefit

## View Examples

Run example configurations:
```powershell
python examples_usage.py
```

Edit the file to try different examples:
- Basic proctoring
- With verification
- Multiple detectors
- Custom configuration
- High performance mode
- High accuracy mode

## Next Steps

### 1. Learn the API
Read [PROCTORING_README.md](PROCTORING_README.md) for full API documentation

### 2. Add Custom Detector
See [modules/phone_detector.py](modules/phone_detector.py) for template

### 3. Understand Architecture
Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design

### 4. Check Transformation Details
Read [TRANSFORMATION_SUMMARY.md](TRANSFORMATION_SUMMARY.md) for what changed

## Getting Help

### Check Logs
The system logs to console. Look for:
```
INFO: Face detector loaded successfully
WARNING: No face detected
ERROR: Failed to load model
```

### Enable Debug Logging
```python
Config.LOG_LEVEL = "DEBUG"
```

### Test Individual Components
```python
# Test camera
from modules import CameraCapture
camera = CameraCapture()
print(camera.start())

# Test face detector
from modules import FaceDetector
detector = FaceDetector()
print(detector.load_model())
```

## Performance Benchmarks

### Expected FPS
| Hardware | FRAME_SKIP=2 | FRAME_SKIP=4 |
|----------|--------------|--------------|
| High-end PC | 25-30 FPS | 28-30 FPS |
| Mid-range | 20-25 FPS | 25-30 FPS |
| Laptop | 15-20 FPS | 20-25 FPS |
| Low-end | 10-15 FPS | 15-20 FPS |

### Processing Time per Frame
| Backend | Average (ms) | Notes |
|---------|--------------|-------|
| opencv | 30-50ms | Fast ✓ |
| mediapipe | 40-60ms | Balanced ✓ |
| ssd | 60-100ms | Accurate |
| mtcnn | 100-150ms | Very accurate |
| retinaface | 150-250ms | Best accuracy |

## Success Indicators

✅ System is working properly if you see:
- Camera feed displaying smoothly
- FPS > 10
- Your face detected with green/orange box
- Console shows "Face detector loaded successfully"
- Report generated on exit

## Demo Script

Quick demo for testing:

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Basic run
python proctor_main.py
# Wave your hand, move around, show you're being tracked
# Press Q to exit

# 3. Check report in console
# Should show:
# - Total frames captured
# - Total frames processed
# - Any alerts (if you moved out of frame)

# 4. Success! ✓
```

## Summary

**To start proctoring in 30 seconds:**
```powershell
pip install -r requirements.txt
python proctor_main.py
```

**To add verification:**
1. Add photos to `data/participants/name/`
2. Set `Config.FACE_VERIFICATION_ENABLED = True`
3. Run `python proctor_main.py`

**To add custom detector:**
1. Copy `modules/phone_detector.py` as template
2. Implement `load_model()` and `process_frame()`
3. Register with `proctor.register_detector()`

---

**You're ready to go!** 🚀

For detailed documentation, see [PROCTORING_README.md](PROCTORING_README.md)
