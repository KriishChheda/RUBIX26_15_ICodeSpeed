"""
Proctoring System - Main Application
Optimized version using MediaPipe + DeepFace pipeline
"""

from modules import ProctorPipeline, ProctorConfig
import cv2


def main():
    """Main entry point for proctoring system"""
    
    # Configure the system
    ProctorConfig.WINDOW_NAME = "AI Proctoring System - Auto-Loading"
    ProctorConfig.FRAME_SKIP = 0  # Process every frame
    ProctorConfig.SHOW_FPS = True
    ProctorConfig.FACE_DETECT_ENABLE = True
    ProctorConfig.FACE_MATCH_ENABLE = True
    ProctorConfig.PARTICIPANT_DATA_PATH = "data/participant.png"
    ProctorConfig.SHOW_ALL_FACE_LANDMARKS = False  # Enable all 478 face mesh points
    ProctorConfig.SHOW_LANDMARK_NUMBERS = False  # Set to True to see landmark numbers
    ProctorConfig.EYE_TRACKING_ENABLE = True  # Enable eye tracking
    ProctorConfig.PHONE_DETECT_ENABLE = True  # Phone detection not yet implemented
    
    print("\n" + "="*70)
    print(" AI PROCTORING SYSTEM - AUTO-LOADING".center(70))
    print("="*70)
    print("\nInitializing proctoring system...")
    
    # Create the proctoring pipeline - detectors auto-load based on config
    proctor = ProctorPipeline(config=ProctorConfig, frame_skip=ProctorConfig.FRAME_SKIP, session_id=None)
    
    # Get session info
    session_info = proctor.session_logger.get_session_summary()
    print(f"\nSession ID: {session_info['session_id']}")
    print(f"Log Directory: {ProctorConfig.PROCTORING_LOG_DIR}")
    
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
    if ProctorConfig.EYE_TRACKING_ENABLE:
        print("  • Press 'c' to calibrate eye tracking")
    print("\nPERFORMANCE:")
    print(f"  • Processing every {ProctorConfig.FRAME_SKIP + 1} frames")
    print(f"  • Camera FPS: {ProctorConfig.CAMERA_FPS}")
    print(f"  • Effective processing rate: ~{ProctorConfig.CAMERA_FPS / (ProctorConfig.FRAME_SKIP + 1):.1f} FPS")
    
    print("\nVISUALIZATION:")
    print(f"  • Face mesh points: {'ALL 478 POINTS' if ProctorConfig.SHOW_ALL_FACE_LANDMARKS else 'KEY POINTS ONLY'}")
    print(f"  • Landmark numbers: {'ENABLED' if ProctorConfig.SHOW_LANDMARK_NUMBERS else 'DISABLED'}")
    print(f"  • Eye tracking: {'ENABLED' if ProctorConfig.EYE_TRACKING_ENABLE else 'DISABLED'}")
    
    if ProctorConfig.FACE_MATCH_ENABLE:
        print("\nFACE VERIFICATION:")
        print(f"  • Status: ENABLED")
        print(f"  • Backend: {ProctorConfig.FACE_MATCHING_BACKEND}")
        print(f"  • Metric: {ProctorConfig.FACE_MATCHING_DISTANCE_METRIC}")
        print(f"  • Threshold: {ProctorConfig.FACE_MATCHING_THRESHOLD}")
        print(f"  • Participant: {ProctorConfig.PARTICIPANT_DATA_PATH}")
    else:
        print("\nFACE VERIFICATION:")
        print(f"  • Status: DISABLED")
    
    print("\nALERTS:")
    print(f"  • Multiple faces: {'ENABLED' if ProctorConfig.ENABLE_MULTI_FACE_ALERT else 'DISABLED'}")
    print(f"  • No face detected: {'ENABLED' if ProctorConfig.ENABLE_NO_FACE_ALERT else 'DISABLED'}")
    
    print("\nLOGGING:")
    print(f"  • Session logs: {session_info['log_file']}")
    print(f"  • Alert logs: {session_info['alerts_file']}")
    
    print("\n" + "="*70)
    print("Starting camera...")
    print("="*70 + "\n")
    
    # Run the proctoring pipeline (uses inherited run() method from CameraPipeline)
    # This handles:
    # - Camera initialization and capture loop
    # - Frame processing (calls process_frame() which we override)
    # - Display with FPS
    # - Exit key detection (q or ESC)
    # - Cleanup in finally block
    try:
        proctor.run()
    except KeyboardInterrupt:
        print("\n\nProctoring session interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Proctoring session ended.")
    print("="*70)


if __name__ == "__main__":
    main()