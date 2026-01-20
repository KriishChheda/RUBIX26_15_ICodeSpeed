"""
Proctoring System - Main Application
Demonstrates the use of ProctorPipeline with multiple detectors
"""

from modules import ProctorPipeline, FaceDetector, Config


def main():
    """Main entry point for proctoring system"""
    
    # Configure the system
    Config.WINDOW_NAME = "AI Proctoring System"
    Config.FRAME_SKIP = 3
    Config.SHOW_FPS = True
    Config.FACE_DETECTION_ENABLED = True
    Config.FACE_DETECTOR_BACKEND = "yolov12s"
    Config.FACE_MODEL_NAME = "Facenet"
    Config.FACE_VERIFICATION_ENABLED = True
    Config.FACE_VERIFICATION_THRESHOLD = 0.5  # More lenient for live camera (0.5-0.6 recommended)
    Config.PARTICIPANT_DATA_PATH = "data/participant.png"

    Config.ENABLE_FACE_VERIFICATION = True
    
    print("\n" + "="*70)
    print(" AI PROCTORING SYSTEM".center(70))
    print("="*70)
    print("\nInitializing proctoring system...")
    
    # Create the proctoring pipeline
    proctor = ProctorPipeline(config=Config, frame_skip=Config.FRAME_SKIP)
    
    # Create and register face detector
    print("\n[1/2] Setting up Face Detector with DeepFace...")
    face_detector = FaceDetector(
        name="FaceDetector",
        enabled=Config.FACE_DETECTION_ENABLED,
        detector_backend=Config.FACE_DETECTOR_BACKEND,
        model_name=Config.FACE_MODEL_NAME,
        verification_threshold=Config.FACE_VERIFICATION_THRESHOLD,
        participant_data_path=Config.PARTICIPANT_DATA_PATH,
        anti_spoofing=True  # Enable anti-spoofing protection
    )
    
    if not proctor.register_detector(face_detector):
        print("ERROR: Failed to register face detector!")
        return
    
    print("✓ Face detector registered successfully")
    
    # You can easily add more detectors here
    # Example:
    # phone_detector = PhoneDetector(name="PhoneDetector", model_path="cv_models/phone_detector")
    # proctor.register_detector(phone_detector)
    
    print("\n[2/2] Starting proctoring session...")
    print("\n" + "-"*70)
    print("SYSTEM STATUS:")
    print("-"*70)
    
    detectors_list = proctor.list_detectors()
    for detector in detectors_list:
        status = "✓ ACTIVE" if detector['enabled'] else "✗ INACTIVE"
        print(f"  {detector['name']}: {status}")
    
    print("-"*70)
    print("\nCONTROLS:")
    print("  • Press 'q' or ESC to quit and generate report")
    print("\nPERFORMANCE:")
    print(f"  • Processing every {Config.FRAME_SKIP + 1} frames")
    print(f"  • Camera FPS: {Config.CAMERA_FPS}")
    print(f"  • Effective processing rate: ~{Config.CAMERA_FPS / (Config.FRAME_SKIP + 1):.1f} FPS")
    
    if Config.FACE_VERIFICATION_ENABLED:
        print("\nVERIFICATION:")
        print(f"  • Face verification: ENABLED")
        print(f"  • Participant data: {Config.PARTICIPANT_DATA_PATH}")
    else:
        print("\nVERIFICATION:")
        print(f"  • Face verification: DISABLED")
        print(f"  • (To enable, set Config.FACE_VERIFICATION_ENABLED = True)")
    
    print("\n" + "="*70)
    print("Starting camera... (First run may download DeepFace models)")
    print("="*70 + "\n")
    
    # Run the proctoring pipeline
    try:
        proctor.run()
    except KeyboardInterrupt:
        print("\n\nProctoring session interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
    
    print("\n" + "="*70)
    print("Proctoring session ended.")
    print("="*70)


if __name__ == "__main__":
    main()