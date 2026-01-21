# Proctoring System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROCTORING SYSTEM                            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              ProctorPipeline (Main)                        │ │
│  │         Inherits from CameraPipeline                       │ │
│  │                                                            │ │
│  │  • Manages multiple detectors                             │ │
│  │  • Frame skipping logic (every Nth frame)                 │ │
│  │  • Alert generation                                       │ │
│  │  • Report generation                                      │ │
│  └────────────┬──────────────────────────────────────────────┘ │
│               │                                                 │
│               │ registers & controls                            │
│               ▼                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Detector Registry                              │  │
│  │                                                          │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌───────────┐ │  │
│  │  │ FaceDetector   │  │ PhoneDetector  │  │  Custom   │ │  │
│  │  │                │  │                │  │ Detector  │ │  │
│  │  │ • DeepFace     │  │ • YOLO/Custom  │  │           │ │  │
│  │  │ • Verification │  │ • Model Path   │  │ (Future)  │ │  │
│  │  │ • Participant  │  │ • Alert on     │  │           │ │  │
│  │  │   Matching     │  │   Detection    │  │           │ │  │
│  │  └────────────────┘  └────────────────┘  └───────────┘ │  │
│  │         ▲                    ▲                  ▲        │  │
│  │         └────────────────────┴──────────────────┘        │  │
│  │                All inherit from:                         │  │
│  │              BaseDetector (Abstract)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Core Components                             │  │
│  │                                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │CameraCapture │  │DisplayWindow │  │    Config    │  │  │
│  │  │              │  │              │  │              │  │  │
│  │  │ • CV2 Input  │  │ • CV2 Display│  │ • Settings   │  │  │
│  │  │ • Resolution │  │ • FPS Display│  │ • Thresholds │  │  │
│  │  │ • FPS Control│  │ • Annotations│  │ • Paths      │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Camera Frame (30 FPS)
      │
      ▼
┌─────────────────┐
│ CameraCapture   │
│ Reads Frame     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  ProctorPipeline.process_frame()   │
│                                     │
│  Frame Counter: 0, 1, 2, 0, 1, 2... │
│                                     │
│  If counter <= FRAME_SKIP:          │
│    → Skip processing, return frame  │
│  Else:                              │
│    → Process with all detectors     │
└────────┬────────────────────────────┘
         │
         ▼  (Process every 3rd frame)
┌─────────────────────────────────────┐
│    For each enabled detector:       │
│                                     │
│  1. FaceDetector.process_frame()    │
│     ├─ Detect faces (DeepFace)      │
│     ├─ Verify identity (optional)   │
│     └─ Draw annotations             │
│                                     │
│  2. PhoneDetector.process_frame()   │
│     ├─ Detect phones                │
│     └─ Draw warnings                │
│                                     │
│  3. [More detectors...]             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Alert Checking                  │
│                                     │
│  • Multiple faces?   → Alert        │
│  • No face?          → Alert        │
│  • Unverified?       → Alert        │
│  • Phone detected?   → Alert        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Add Overlay Info                 │
│                                     │
│  • Detector status                  │
│  • Verification result              │
│  • Frame count                      │
│  • FPS                              │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ DisplayWindow   │
│ Show Frame      │
└─────────────────┘
```

## Face Detection Flow (DeepFace)

```
Input Frame
     │
     ▼
┌─────────────────────────────────────┐
│  DeepFace.extract_faces()           │
│                                     │
│  Backend Options:                   │
│  • opencv      (fast)               │
│  • mediapipe   (fast + accurate)    │
│  • mtcnn       (accurate)           │
│  • retinaface  (most accurate)      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Face Objects                       │
│  [{                                 │
│    facial_area: {x, y, w, h},       │
│    confidence: 0.95                 │
│  }]                                 │
└────────┬────────────────────────────┘
         │
         ├─ If verification enabled
         │
         ▼
┌─────────────────────────────────────┐
│  DeepFace.verify()                  │
│                                     │
│  Compare against:                   │
│  data/participants/                 │
│    ├─ john_doe/                     │
│    │   ├─ photo1.jpg                │
│    │   └─ photo2.jpg                │
│    └─ jane_smith/                   │
│        └─ photo1.jpg                │
│                                     │
│  Model Options:                     │
│  • Facenet   (recommended)          │
│  • VGG-Face                         │
│  • ArcFace   (most accurate)        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Verification Result                │
│  {                                  │
│    verified: true/false,            │
│    participant_id: "john_doe",      │
│    distance: 0.23,                  │
│    threshold: 0.40                  │
│  }                                  │
└────────┬────────────────────────────┘
         │
         ▼
    Draw Boxes
    Green = Verified
    Red = Not Verified
```

## Adding New Detector

```
Step 1: Create Detector Class
┌─────────────────────────────────────┐
│  class MyDetector(BaseDetector):    │
│                                     │
│      def load_model(self):          │
│          # Load model               │
│          self.model = ...           │
│          self.initialized = True    │
│          return True                │
│                                     │
│      def process_frame(self, frame):│
│          # Your logic here          │
│          detections = ...           │
│          return frame, results      │
└─────────────────────────────────────┘
         │
         ▼
Step 2: Register with Pipeline
┌─────────────────────────────────────┐
│  proctor = ProctorPipeline()        │
│  detector = MyDetector()            │
│  proctor.register_detector(detector)│
└─────────────────────────────────────┘
         │
         ▼
Step 3: Run
┌─────────────────────────────────────┐
│  proctor.run()                      │
│                                     │
│  Your detector is automatically     │
│  called for each processed frame    │
└─────────────────────────────────────┘
```

## File Structure

```
HD_ML_stuff/
│
├── modules/
│   ├── base_detector.py       ← Abstract base for all detectors
│   ├── proctor_pipeline.py    ← Main proctoring logic
│   ├── face_detector.py       ← DeepFace implementation
│   ├── phone_detector.py      ← Template for custom detectors
│   ├── camera_pipeline.py     ← Base pipeline (inherited)
│   ├── camera_input.py        ← Camera handling
│   ├── display.py             ← Display handling
│   └── config.py              ← Configuration
│
├── cv_models/                 ← Place custom models here
│   ├── phone_detector/
│   └── [other models]
│
├── data/
│   └── participants/          ← Participant photos
│       ├── participant_1/
│       └── participant_2/
│
├── proctor_main.py            ← Main application
├── examples_usage.py          ← Usage examples
└── requirements.txt           ← Dependencies
```

## Configuration Hierarchy

```
Config Class (config.py)
│
├── Camera Settings
│   ├── CAMERA_ID = 0
│   ├── CAMERA_WIDTH = 1080
│   ├── CAMERA_HEIGHT = 720
│   └── CAMERA_FPS = 30
│
├── Performance Settings
│   ├── FRAME_SKIP = 2          ← Process every 3rd frame
│   └── MAX_FPS = 60
│
├── Face Detection (DeepFace)
│   ├── FACE_DETECTOR_BACKEND = "opencv"
│   ├── FACE_MODEL_NAME = "Facenet"
│   ├── FACE_VERIFICATION_THRESHOLD = 0.4
│   └── FACE_VERIFICATION_ENABLED = False
│
├── Proctoring
│   ├── PARTICIPANT_DATA_PATH = "data/participants"
│   ├── CV_MODELS_PATH = "cv_models"
│   ├── ENABLE_MULTI_FACE_ALERT = True
│   └── ENABLE_NO_FACE_ALERT = True
│
└── Display Settings
    ├── WINDOW_NAME = "Proctoring System"
    ├── SHOW_FPS = True
    └── FULLSCREEN = False
```

## Performance Optimization

```
30 FPS Camera
     │
     ├─ FRAME_SKIP = 0 (No skip)
     │  → Process all 30 frames
     │  → ~30 FPS processing
     │  → HIGH CPU usage
     │
     ├─ FRAME_SKIP = 1 (Every 2nd)
     │  → Process 15 frames
     │  → ~15 FPS processing
     │  → MEDIUM CPU usage
     │
     ├─ FRAME_SKIP = 2 (Every 3rd) ✓ RECOMMENDED
     │  → Process 10 frames
     │  → ~10 FPS processing
     │  → LOW CPU usage
     │  → Still excellent for proctoring
     │
     └─ FRAME_SKIP = 4 (Every 5th)
        → Process 6 frames
        → ~6 FPS processing
        → VERY LOW CPU usage
        → Good for low-power devices
```

## Alert System

```
Detection Results
      │
      ▼
┌─────────────────────────────────────┐
│  ProctorPipeline._check_alerts()   │
│                                     │
│  Checks:                            │
│  • Multiple faces? (> 1)            │
│  • No face? (== 0)                  │
│  • Unverified person?               │
│  • Phone detected?                  │
│  • [Custom conditions...]           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Alert Created                      │
│  {                                  │
│    timestamp: 1234567890,           │
│    detector: "FaceDetector",        │
│    type: "multiple_faces",          │
│    message: "Multiple faces: 2",    │
│    severity: "warning"              │
│  }                                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Add to proctoring_results.alerts  │
│  Log to console                     │
└─────────────────────────────────────┘
```

## Report Generation

```
Session End
     │
     ▼
┌─────────────────────────────────────┐
│  ProctorPipeline.get_proctoring_    │
│  report()                           │
│                                     │
│  Aggregates:                        │
│  • Total frames captured            │
│  • Total frames processed           │
│  • Processing ratio                 │
│  • All alerts                       │
│  • Alert summary by type            │
│  • Detector statistics              │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Report Output                      │
│                                     │
│  =====================================│
│  PROCTORING SESSION REPORT          │
│  =====================================│
│  Total Frames: 3000                 │
│  Processed: 1000 (33.33%)           │
│  Alerts: 15                         │
│                                     │
│  Alert Summary:                     │
│    - multiple_faces: 5              │
│    - no_face: 10                    │
│  =====================================│
└─────────────────────────────────────┘
```
