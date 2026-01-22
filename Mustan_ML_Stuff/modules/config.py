"""
Configuration Module
Central configuration for camera pipeline and proctoring system
"""


class ProctorConfig:
    """Pipeline configuration settings"""
    
    # Camera Settings
    CAMERA_ID = 0
    CAMERA_WIDTH = 1080
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Display Settings
    WINDOW_NAME = "Proctoring System"
    FULLSCREEN = False
    SHOW_FPS = True
    DISPLAY_FEED = True  # When False, no frames are displayed (background mode)
    
    # Pipeline Settings
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Performance Settings
    MAX_FPS = 60  # Maximum FPS to process
    FRAME_SKIP = 2  # Process every 3rd frame (0 = process all, 1 = every 2nd, 2 = every 3rd)
    
    # Shared Memory Settings (for frontend frame streaming)
    SHARED_MEMORY_ENABLED = True  # Enable shared memory buffer for zero-copy frame sharing
    SHARED_MEMORY_PATH = "shared_memory/proctor_frame.mmap"  # Relative to project root
    
    # Face Detection Settings (MediaPipe)
    FACE_DETECT_ENABLE = True
    FACE_MODEL_SELECTION = 1  # 0 for short-range (<2m), 1 for full-range (<5m)
    FACE_MIN_DETECTION_CONFIDENCE = 0.7  # Minimum confidence for face detection
    FACE_MIN_TRACKING_CONFIDENCE = 0.5  # Minimum confidence for face tracking
    
    # Face Mesh Visualization Settings
    SHOW_ALL_FACE_LANDMARKS = True  # Show all 478 face mesh points (set False for key points only)
    SHOW_LANDMARK_NUMBERS = False  # Show landmark index numbers (Warning: lots of text!)
    
    # Eye Tracking Settings (MediaPipe)
    EYE_TRACKING_ENABLE = True  # Enable eye movement detection and tracking (MediaPipe-based)
    
    # Eye Detection Thresholds
    EYE_HORIZONTAL_THRESHOLD = 0.30  # Looking left/right threshold (iris position ratio)
    EYE_VERTICAL_DOWN_THRESHOLD = 0.10  # Looking down threshold (iris position ratio)
    EYE_VERTICAL_UP_THRESHOLD = -0.30  # Looking up threshold (iris position ratio)
    EYE_CLOSED_THRESHOLD = 0.15  # Eye Aspect Ratio (EAR) threshold for closed eyes
    
    # Eye Tracking MediaPipe Settings
    EYE_FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection
    EYE_FACE_PRESENCE_CONFIDENCE = 0.5  # Minimum confidence for face presence
    EYE_TRACKING_CONFIDENCE = 0.5  # Minimum confidence for tracking
    
    # Eye Tracking Model
    EYE_TRACKING_MODEL_PATH = "cv_models/face_landmarker.task"  # MediaPipe face landmarker model
    EYE_TRACKING_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    # Eye Tracking Visualization
    EYE_DRAW_LANDMARKS = True  # Draw eye bounding boxes and iris tracking
    EYE_DEBUG_MODE = False  # Show ratios and thresholds on screen
    
    # Eye Tracking Alerts
    EYE_ENABLE_LOOKING_DOWN_ALERT = True  # Alert when looking down
    EYE_ENABLE_LOOKING_AWAY_ALERT = True  # Alert when looking left/right
    EYE_ENABLE_NO_FACE_ALERT = True  # Alert when no face detected for eye tracking
    
    # Face Matching Settings (DeepFace)
    FACE_MATCH_ENABLE = True  # Enable face verification against participant
    FACE_MATCHING_BACKEND = "Facenet"  # DeepFace backend: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace, Dlib, SFace
    FACE_MATCHING_DISTANCE_METRIC = "cosine"  # Distance metric: cosine, euclidean, euclidean_l2
    FACE_MATCHING_THRESHOLD = 0.5  # Distance threshold (model-specific, lower = stricter)
    # Recommended thresholds (cosine): VGG-Face=0.40, Facenet=0.40, Facenet512=0.30, ArcFace=0.68, Dlib=0.07, SFace=0.593, OpenFace=0.10
    
    # Phone Detection Settings
    PHONE_DETECT_ENABLE = True  # Enable phone detection (requires phone detector model)
    PHONE_MODEL_PATH = "cv_models/phone.pt"  # Path to phone detection model
    PHONE_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold for phone detection
    
    # Proctoring Settings
    PARTICIPANT_DATA_PATH = "data/participant.png"  # Single participant reference image
    CV_MODELS_PATH = "cv_models"
    FACE_MARKER_MODEL_PATH = "cv_models/face_landmarker.task"  # MediaPipe Face Landmarker model path
    ENABLE_MULTI_FACE_ALERT = True  # Alert when multiple faces detected
    ENABLE_NO_FACE_ALERT = True  # Alert when no face detected
    
    # Session Logging Settings
    PROCTORING_LOG_DIR = "logs/proctoring"  # Directory for session logs
    EYE_MOVEMENT_LOG_DIR = "logs/eye_movements"  # Directory for eye movement logs
    SAVE_SESSION_LOGS = True  # Save session logs to file
    LOG_ALERTS_ONLY = True  # Only log alert messages (no info/debug to console)
    
    # Alert Communication Settings
    ALERT_STATE_FILE_NAME = "alert_state.txt"  # File name for alert state communication
    ALERT_COOLDOWN_SECONDS = 5  # Minimum time between same alert types
    
    @classmethod
    def from_dict(cls, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
    
    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary"""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and key.isupper()
        }
