"""
Proctoring System - Background Mode Example
Runs proctoring in background without displaying frames
"""

from modules import ProctorPipeline, ProctorConfig
import cv2
import time


def main():
    """Main entry point for background proctoring"""
    
    # Configure the system for BACKGROUND MODE
    ProctorConfig.WINDOW_NAME = "AI Proctoring System - Background Mode"
    ProctorConfig.FRAME_SKIP = 0  # Process every frame
    ProctorConfig.SHOW_FPS = True
    ProctorConfig.DISPLAY_FEED = False  # *** DISABLE DISPLAY - BACKGROUND MODE ***
    ProctorConfig.FACE_DETECT_ENABLE = True
    ProctorConfig.FACE_MATCH_ENABLE = True
    ProctorConfig.PARTICIPANT_DATA_PATH = "data/participant.png"
    ProctorConfig.SHOW_ALL_FACE_LANDMARKS = False
    ProctorConfig.SHOW_LANDMARK_NUMBERS = False
    ProctorConfig.EYE_TRACKING_ENABLE = True
    ProctorConfig.PHONE_DETECT_ENABLE = False

    print("\n" + "="*70)
    print(" AI PROCTORING SYSTEM - BACKGROUND MODE".center(70))
    print("="*70)
    print("\n⚠️  DISPLAY IS DISABLED - RUNNING IN BACKGROUND")
    print("   Webcam feed will be processed without showing frames")
    print("   All alerts and logs will be saved to disk\n")
    
    print("Initializing proctoring system...")
    
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
    print("\nMODE:")
    print(f"  • Display Feed: {'ENABLED' if ProctorConfig.DISPLAY_FEED else '⚠️  DISABLED (Background Mode)'}")
    print("\nCONTROLS:")
    print("  • Press CTRL+C to stop and generate report")
    
    print("\nPERFORMANCE:")
    print(f"  • Processing every {ProctorConfig.FRAME_SKIP + 1} frames")
    print(f"  • Camera FPS: {ProctorConfig.CAMERA_FPS}")
    print(f"  • Effective processing rate: ~{ProctorConfig.CAMERA_FPS / (ProctorConfig.FRAME_SKIP + 1):.1f} FPS")
    
    if ProctorConfig.FACE_MATCH_ENABLE:
        print("\nFACE VERIFICATION:")
        print(f"  • Status: ENABLED")
        print(f"  • Backend: {ProctorConfig.FACE_MATCHING_BACKEND}")
        print(f"  • Metric: {ProctorConfig.FACE_MATCHING_DISTANCE_METRIC}")
        print(f"  • Threshold: {ProctorConfig.FACE_MATCHING_THRESHOLD}")
        print(f"  • Participant: {ProctorConfig.PARTICIPANT_DATA_PATH}")
    
    print("\nALERTS:")
    print(f"  • Multiple faces: {'ENABLED' if ProctorConfig.ENABLE_MULTI_FACE_ALERT else 'DISABLED'}")
    print(f"  • No face detected: {'ENABLED' if ProctorConfig.ENABLE_NO_FACE_ALERT else 'DISABLED'}")
    
    print("\nLOGGING:")
    print(f"  • Session logs: {session_info['log_file']}")
    print(f"  • Alert logs: {session_info['alerts_file']}")
    
    print("\n" + "="*70)
    print("Starting background proctoring...")
    print("="*70 + "\n")
    
    # Track start time for demo purposes (could run indefinitely)
    start_time = time.time()
    
    # Run the proctoring pipeline
    try:
        proctor.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proctoring session interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"Proctoring session ended after {elapsed:.1f} seconds.")
    print("="*70)


if __name__ == "__main__":
    main()
