"""
Pipeline Module
Base camera pipeline class

YOLOv8 Face Detection Pipeline

"""
import cv2
import logging
import time
from .camera_input import CameraCapture
from .display import DisplayWindow
from .config import ProctorConfig

class CameraPipeline:
    """Main pipeline class that orchestrates camera capture and display"""
    
    def __init__(self, config=None):
        """
        Initialize the camera pipeline
        
        Args:
            config: Configuration object (uses default ProctorConfig if None)
        """
        self.config = config or ProctorConfig()
        self.camera = None
        self.display = None
        self.is_running = False
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the pipeline"""
        if self.config.ENABLE_LOGGING:
            logging.basicConfig(
                level=getattr(logging, self.config.LOG_LEVEL),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize camera and display components"""
        logging.info("Initializing camera pipeline...")
        
        # Initialize camera
        self.camera = CameraCapture(
            camera_id=self.config.CAMERA_ID,
            width=self.config.CAMERA_WIDTH,
            height=self.config.CAMERA_HEIGHT,
            fps=self.config.CAMERA_FPS
        )
        
        if not self.camera.start():
            logging.error("Failed to start camera")
            return False
        
        # Initialize display only if DISPLAY_FEED is enabled
        if getattr(self.config, 'DISPLAY_FEED', True):
            self.display = DisplayWindow(
                window_name=self.config.WINDOW_NAME,
                fullscreen=self.config.FULLSCREEN
            )
            
            if not self.display.create_window():
                logging.error("Failed to create display window")
                self.camera.stop()
                return False
        else:
            self.display = None
            logging.info("Display disabled (background mode)")
        
        logging.info("Pipeline initialized successfully")
        return True
    
    def process_frame(self, frame):
        """
        Process a single frame (override this method for custom processing)
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame
        """
        # Default: no processing, just return the frame
        return frame
    
    def run(self):
        """Main pipeline loop"""
        if not self.initialize():
            logging.error("Pipeline initialization failed")
            return
        
        self.is_running = True
        logging.info("Starting pipeline loop...")
        
        # FPS tracking
        fps = 0
        frame_count = 0
        start_time = time.time()
        frame_time = 1.0 / self.config.MAX_FPS if self.config.MAX_FPS > 0 else 0
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Read frame from camera
                success, frame = self.camera.read_frame()
                
                if not success:
                    logging.warning("Failed to read frame, continuing...")
                    continue
                
                # Process frame (can be overridden)
                processed_frame = self.process_frame(frame)
                
                # Display frame only if DISPLAY_FEED is enabled
                if self.display:
                    if self.config.SHOW_FPS:
                        self.display.show_frame(processed_frame, fps=fps)
                    else:
                        self.display.show_frame(processed_frame)
                    
                    # Check for exit key
                    if self.display.check_exit_key(1):
                        logging.info("Exit key pressed")
                        break
                else:
                    # In background mode, check for keyboard interrupt or use time-based exit
                    # Small delay to prevent CPU spinning
                    time.sleep(0.001)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                
                # Frame rate limiting
                if frame_time > 0:
                    processing_time = time.time() - loop_start
                    sleep_time = frame_time - processing_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logging.info("Pipeline interrupted by user")
        
        except Exception as e:
            logging.error(f"Pipeline error: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[DEBUG] === ENTERING CameraPipeline.cleanup() ===")
        print("[DEBUG] CameraPipeline Step 0: Starting cleanup")
        logging.info("Cleaning up pipeline resources...")
        self.is_running = False
        
        try:
            print("[DEBUG] CameraPipeline Step 1: Checking camera")
            if self.camera:
                print("[DEBUG] CameraPipeline Step 2: Stopping camera")
                self.camera.stop()
                print("[DEBUG] CameraPipeline Step 3: Camera stopped")
        except Exception as e:
            print(f"[DEBUG] ERROR stopping camera: {e}")
            logging.error(f"Error stopping camera: {e}")
        
        try:
            print("[DEBUG] CameraPipeline Step 4: Checking display")
            if self.display:
                print("[DEBUG] CameraPipeline Step 5: Destroying display")
                self.display.destroy_all()
                print("[DEBUG] CameraPipeline Step 6: Display destroyed")
        except Exception as e:
            print(f"[DEBUG] ERROR destroying display: {e}")
            logging.error(f"Error destroying display: {e}")
        
        print("[DEBUG] CameraPipeline Step 7: Cleanup complete")
        logging.info("Pipeline cleanup complete")
        print("[DEBUG] === EXITING CameraPipeline.cleanup() ===")