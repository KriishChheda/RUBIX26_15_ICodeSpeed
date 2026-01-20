"""
Configuration Module
Central configuration for camera pipeline and proctoring system
"""


class Config:
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
    
    # Pipeline Settings
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Performance Settings
    MAX_FPS = 60  # Maximum FPS to process
    FRAME_SKIP = 2  # Process every 3rd frame (0 = process all, 1 = every 2nd, 2 = every 3rd)
    
    # Face Detection Settings (DeepFace)
    FACE_DETECTION_ENABLED = True
    FACE_DETECTOR_BACKEND = "opencv"  # 'opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe'
    FACE_MODEL_NAME = "Facenet"  # 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace'
    FACE_DISTANCE_METRIC = "cosine"  # 'cosine', 'euclidean', 'euclidean_l2'
    FACE_VERIFICATION_THRESHOLD = 0.5  # Lower = stricter (0.4 for images, 0.5-0.6 for live camera)
    FACE_VERIFICATION_ENABLED = True  # Enable face verification against known participants
    
    # Proctoring Settings
    PARTICIPANT_DATA_PATH = "data/participant.png"  # Single participant reference image
    CV_MODELS_PATH = "cv_models"
    ENABLE_FACE_VERIFICATION = False  # Verify against known participant faces
    ENABLE_MULTI_FACE_ALERT = True  # Alert when multiple faces detected
    ENABLE_NO_FACE_ALERT = True  # Alert when no face detected
    
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
