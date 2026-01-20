"""
Configuration Module
Central configuration for eye movement detection pipeline
"""


class Config:
    """Pipeline configuration settings"""
    
    # Camera Settings
    CAMERA_ID = 0
    CAMERA_WIDTH = 640   # Reduced resolution for performance
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Display Settings
    WINDOW_NAME = "Eye Movement Detection Pipeline"
    FULLSCREEN = False
    SHOW_FPS = True
    
    # Pipeline Settings
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Performance Settings
    MAX_FPS = 30       # Cap FPS to match camera
    FRAME_SKIP = 2     # Process every 3rd frame (skip 2) to reduce load
    
    # Eye Detection Settings
    EYE_DETECTION_ENABLED = True
    EYE_MODEL_PATH = "best.pt"  # YOLOv8 model with keypoints (best.pt)
    EYE_CONFIDENCE_THRESHOLD = 0.5
    EYE_BOX_COLOR = (255, 0, 0)  # Blue (B, G, R)
    EYE_BOX_THICKNESS = 2
    EYE_SHOW_CONFIDENCE = True
    EYE_SHOW_LABEL = True
    EYE_SHOW_KEYPOINTS = True  # Show eye keypoints (inner, outer, pupil)
    
    # Face detection (optional - best.pt detects eyes directly)
    FACE_DETECTION_ENABLED = False  # Not needed with best.pt
    FACE_MODEL_NAME = "yolov8n-face.pt"  # YOLOv8 face model
    FACE_CONFIDENCE_THRESHOLD = 0.5
    
    # Risk Analysis Settings (based on pupil position geometry)
    RISK_VERTICAL_THRESHOLD = 0.15  # Threshold for looking up/down
    RISK_HORIZONTAL_MIN = 0.3  # Minimum horizontal ratio for center
    RISK_HORIZONTAL_MAX = 0.7  # Maximum horizontal ratio for center
    SHOW_RISK_STATUS = True  # Display risk status on frame
    
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
