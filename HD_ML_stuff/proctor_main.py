"""
Proctoring System - Main Application
Optimized version using MediaPipe + DeepFace pipeline
"""

from modules import ProctorPipeline, FaceDetector, FaceMatcher, Config
import cv2


def main():
    """Main entry point for proctoring system"""
    
    # Configure the system
    Config.WINDOW_NAME = "AI Proctoring System - DeepFace Optimized"
    Config.FRAME_SKIP = 2  # Process every 3rd frame
    Config.SHOW_FPS = True
    Config.FACE_DETECTION_ENABLED = True
    Config.FACE_MATCHING_ENABLED = False
    Config.PARTICIPANT_DATA_PATH = "data/participant.png"
    
    print("\n" + "="*70)
    print(" AI PROCTORING SYSTEM - DEEPFACE OPTIMIZED".center(70))
    print("="*70)
    print("\nInitializing proctoring system...")
    
    # Create the proctoring pipeline with session logging
    proctor = ProctorPipeline(config=Config, frame_skip=Config.FRAME_SKIP, session_id=None)
    
    # Get session info
    session_info = proctor.session_logger.get_session_summary()
    print(f"\nSession ID: {session_info['session_id']}")
    print(f"Log Directory: {Config.PROCTORING_LOG_DIR}")
    
    # Create and register face detector (MediaPipe)
    print("\n[1/2] Setting up Face Detector (MediaPipe)...")
    face_detector = FaceDetector(
        name="FaceDetector",
        enabled=Config.FACE_DETECTION_ENABLED,
        model_selection=Config.FACE_MODEL_SELECTION,
        min_detection_confidence=Config.FACE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=Config.FACE_MIN_TRACKING_CONFIDENCE
    )
    
    if not proctor.register_detector(face_detector):
        print("ERROR: Failed to register face detector!")
        return
    
    print("✓ Face detector registered successfully")
    
    # Create and register face matcher (DeepFace with configurable backend)
    if Config.FACE_MATCHING_ENABLED:
        print(f"\n[2/2] Setting up Face Matcher (DeepFace - {Config.FACE_MATCHING_BACKEND})...")
        face_matcher = FaceMatcher(
            name="FaceMatcher",
            enabled=True,
            model_name=Config.FACE_MATCHING_BACKEND,
            distance_metric=Config.FACE_MATCHING_DISTANCE_METRIC,
            distance_threshold=Config.FACE_MATCHING_THRESHOLD,
            participant_image_path=Config.PARTICIPANT_DATA_PATH
        )
        
        if not proctor.register_detector(face_matcher):
            print("ERROR: Failed to register face matcher!")
            return
        
        print("✓ Face matcher registered successfully")
    
    # You can easily add more detectors here
    # Example:
    # eye_detector = EyeMovementDetector(model_path="cv_models/best.pt")
    # proctor.register_detector(eye_detector)
    
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
    
    if Config.FACE_MATCHING_ENABLED:
        print("\nFACE VERIFICATION:")
        print(f"  • Status: ENABLED")
        print(f"  • Backend: {Config.FACE_MATCHING_BACKEND}")
        print(f"  • Metric: {Config.FACE_MATCHING_DISTANCE_METRIC}")
        print(f"  • Threshold: {Config.FACE_MATCHING_THRESHOLD}")
        print(f"  • Participant: {Config.PARTICIPANT_DATA_PATH}")
    else:
        print("\nFACE VERIFICATION:")
        print(f"  • Status: DISABLED")
    
    print("\nALERTS:")
    print(f"  • Multiple faces: {'ENABLED' if Config.ENABLE_MULTI_FACE_ALERT else 'DISABLED'}")
    print(f"  • No face detected: {'ENABLED' if Config.ENABLE_NO_FACE_ALERT else 'DISABLED'}")
    
    print("\nLOGGING:")
    print(f"  • Session logs: {session_info['log_file']}")
    print(f"  • Alert logs: {session_info['alerts_file']}")
    
    print("\n" + "="*70)
    print("Starting camera...")
    print("="*70 + "\n")
    
    # Run the proctoring pipeline
    try:
        if not proctor.start():
            print("ERROR: Failed to start proctoring pipeline")
            return
        
        while True:
            # Capture frame
            frame = proctor.capture_frame()
            if frame is None:
                break
            
            # Process frame with optimized pipeline
            processed_frame = proctor.process_frame(frame)
            
            # Display
            proctor.display_frame(processed_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("\nQuitting...")
                break
        
    except KeyboardInterrupt:
        print("\n\nProctoring session interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup and save session
        proctor.stop()
    
    print("\n" + "="*70)
    print("Proctoring session ended.")
    print("="*70)


if __name__ == "__main__":
    main()