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
    
    # Face Detection Settings (MediaPipe)
    FACE_DETECT_ENABLE = True
    FACE_MODEL_SELECTION = 1  # 0 for short-range (<2m), 1 for full-range (<5m)
    FACE_MIN_DETECTION_CONFIDENCE = 0.7  # Minimum confidence for face detection
    FACE_MIN_TRACKING_CONFIDENCE = 0.5  # Minimum confidence for face tracking
    
    # Face Mesh Visualization Settings
    SHOW_ALL_FACE_LANDMARKS = True  # Show all 478 face mesh points (set False for key points only)
    SHOW_LANDMARK_NUMBERS = False  # Show landmark index numbers (Warning: lots of text!)
    
    # Eye Tracking Settings
    EYE_TRACKING_ENABLE = True  # Enable eye movement detection and tracking (MediaPipe-based)
    
    # Face Matching Settings (DeepFace)
    FACE_MATCH_ENABLE = True  # Enable face verification against participant
    FACE_MATCHING_BACKEND = "Facenet"  # DeepFace backend: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace, Dlib, SFace
    FACE_MATCHING_DISTANCE_METRIC = "cosine"  # Distance metric: cosine, euclidean, euclidean_l2
    FACE_MATCHING_THRESHOLD = 0.5  # Distance threshold (model-specific, lower = stricter)
    # Recommended thresholds (cosine): VGG-Face=0.40, Facenet=0.40, Facenet512=0.30, ArcFace=0.68, Dlib=0.07, SFace=0.593, OpenFace=0.10
    
    # Phone Detection Settings
    PHONE_DETECT_ENABLE = False  # Enable phone detection (requires phone detector model)
    PHONE_MODEL_PATH = "cv_models/phone_detector.pt"  # Path to phone detection model
    
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
    
    # Alert Settings
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
