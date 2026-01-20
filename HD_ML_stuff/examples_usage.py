"""
Example: Advanced Proctoring System Usage
Shows various ways to configure and use the proctoring system
"""

from modules import ProctorPipeline, FaceDetector, Config
from modules.phone_detector import PhoneDetector  # Example custom detector


def example_basic_proctoring():
    """Basic proctoring with face detection only"""
    print("Example 1: Basic Proctoring\n")
    
    # Create pipeline
    proctor = ProctorPipeline(frame_skip=2)
    
    # Add face detector
    face_detector = FaceDetector(name="FaceDetector")
    proctor.register_detector(face_detector)
    
    # Run
    proctor.run()


def example_with_verification():
    """Proctoring with face verification against known participants"""
    print("Example 2: Proctoring with Face Verification\n")
    
    Config.FACE_VERIFICATION_ENABLED = True
    
    # Create pipeline
    proctor = ProctorPipeline(frame_skip=2)
    
    # Add face detector with verification enabled
    face_detector = FaceDetector(
        name="FaceDetector",
        detector_backend="opencv",
        model_name="Facenet",
        participant_data_path="data/participants"
    )
    
    proctor.register_detector(face_detector)
    
    # Run
    proctor.run()
    
    # Get report
    report = proctor.get_proctoring_report()
    print("\nSession Report:")
    print(f"Frames Processed: {report['session_stats']['total_frames_processed']}")
    print(f"Alerts: {len(report['alerts'])}")


def example_multiple_detectors():
    """Proctoring with multiple detection modules"""
    print("Example 3: Multiple Detectors\n")
    
    # Create pipeline
    proctor = ProctorPipeline(frame_skip=2)
    
    # Add face detector
    face_detector = FaceDetector(name="FaceDetector")
    proctor.register_detector(face_detector)
    
    # Add phone detector (example - needs actual model)
    # phone_detector = PhoneDetector(
    #     name="PhoneDetector",
    #     model_path="cv_models/phone_detector"
    # )
    # proctor.register_detector(phone_detector)
    
    # List all detectors
    print("Registered Detectors:")
    for detector in proctor.list_detectors():
        print(f"  - {detector['name']}: {'Enabled' if detector['enabled'] else 'Disabled'}")
    
    # Run
    proctor.run()


def example_custom_configuration():
    """Custom configuration example"""
    print("Example 4: Custom Configuration\n")
    
    # Custom config
    Config.CAMERA_WIDTH = 640
    Config.CAMERA_HEIGHT = 480
    Config.FRAME_SKIP = 1  # Process every 2nd frame
    Config.FACE_DETECTOR_BACKEND = "mediapipe"  # Fast and accurate
    Config.FACE_MODEL_NAME = "ArcFace"  # Different recognition model
    
    # Create pipeline
    proctor = ProctorPipeline(config=Config, frame_skip=Config.FRAME_SKIP)
    
    # Add detector
    face_detector = FaceDetector(
        detector_backend=Config.FACE_DETECTOR_BACKEND,
        model_name=Config.FACE_MODEL_NAME
    )
    proctor.register_detector(face_detector)
    
    # Run
    proctor.run()


def example_dynamic_detector_control():
    """Example of enabling/disabling detectors dynamically"""
    print("Example 5: Dynamic Detector Control\n")
    
    # Create pipeline
    proctor = ProctorPipeline(frame_skip=2)
    
    # Add multiple detectors
    face_detector = FaceDetector(name="FaceDetector")
    proctor.register_detector(face_detector)
    
    # Initially disable face detector
    proctor.disable_detector("FaceDetector")
    
    print("Initial state:")
    for detector in proctor.list_detectors():
        print(f"  {detector['name']}: {detector['enabled']}")
    
    # Enable it back
    proctor.enable_detector("FaceDetector")
    
    print("\nAfter enabling:")
    for detector in proctor.list_detectors():
        print(f"  {detector['name']}: {detector['enabled']}")
    
    # Run
    proctor.run()


def example_high_performance():
    """High performance configuration for slower systems"""
    print("Example 6: High Performance Mode\n")
    
    # Optimize for performance
    Config.CAMERA_WIDTH = 640
    Config.CAMERA_HEIGHT = 480
    Config.FRAME_SKIP = 4  # Process every 5th frame
    Config.FACE_DETECTOR_BACKEND = "opencv"  # Fastest backend
    
    proctor = ProctorPipeline(frame_skip=Config.FRAME_SKIP)
    
    face_detector = FaceDetector(
        detector_backend=Config.FACE_DETECTOR_BACKEND
    )
    proctor.register_detector(face_detector)
    
    print(f"Processing every {Config.FRAME_SKIP + 1} frames")
    print(f"Expected processing rate: ~{Config.CAMERA_FPS / (Config.FRAME_SKIP + 1):.1f} FPS")
    
    proctor.run()


def example_high_accuracy():
    """High accuracy configuration"""
    print("Example 7: High Accuracy Mode\n")
    
    # Optimize for accuracy
    Config.CAMERA_WIDTH = 1920
    Config.CAMERA_HEIGHT = 1080
    Config.FRAME_SKIP = 1  # Process more frames
    Config.FACE_DETECTOR_BACKEND = "retinaface"  # Most accurate
    Config.FACE_MODEL_NAME = "ArcFace"  # Best recognition model
    Config.FACE_VERIFICATION_THRESHOLD = 0.3  # Stricter verification
    
    proctor = ProctorPipeline(frame_skip=Config.FRAME_SKIP)
    
    face_detector = FaceDetector(
        detector_backend=Config.FACE_DETECTOR_BACKEND,
        model_name=Config.FACE_MODEL_NAME,
        verification_threshold=Config.FACE_VERIFICATION_THRESHOLD
    )
    proctor.register_detector(face_detector)
    
    print("High accuracy mode enabled")
    print(f"Detector: {Config.FACE_DETECTOR_BACKEND}")
    print(f"Model: {Config.FACE_MODEL_NAME}")
    
    proctor.run()


if __name__ == "__main__":
    print("="*70)
    print(" AI PROCTORING SYSTEM - USAGE EXAMPLES".center(70))
    print("="*70)
    print("\nAvailable examples:")
    print("  1. Basic proctoring")
    print("  2. Proctoring with verification")
    print("  3. Multiple detectors")
    print("  4. Custom configuration")
    print("  5. Dynamic detector control")
    print("  6. High performance mode")
    print("  7. High accuracy mode")
    print("\nEdit this file to run different examples")
    print("="*70 + "\n")
    
    # Run the basic example by default
    # Change this to run different examples
    example_basic_proctoring()
    
    # Uncomment to run other examples:
    # example_with_verification()
    # example_multiple_detectors()
    # example_custom_configuration()
    # example_dynamic_detector_control()
    # example_high_performance()
    # example_high_accuracy()
