"""
Camera Input Module
Handles camera capture and frame retrieval
"""

import cv2
import logging


class CameraCapture:
    """Handles camera input and frame capture"""
    
    def __init__(self, camera_id=0, width=None, height=None, fps=None):
        """
        Initialize camera capture
        
        Args:
            camera_id: Camera device ID (default: 0)
            width: Frame width (optional)
            height: Frame height (optional)
            fps: Frames per second (optional)
        """
        self.camera_id = camera_id
        self.capture = None
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False
        
        logging.info(f"Initializing camera with ID: {camera_id}")
        
    def start(self):
        """Start camera capture"""
        try:
            self.capture = cv2.VideoCapture(self.camera_id)
            
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Set camera properties if specified
            if self.width:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps:
                self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_opened = True
            logging.info("Camera started successfully")
            
            # Log actual camera properties
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            logging.info(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            return True
            
        except Exception as e:
            logging.error(f"Error starting camera: {e}")
            return False
    
    def read_frame(self):
        """
        Read a frame from the camera
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the image
        """
        if not self.is_opened or self.capture is None:
            logging.warning("Camera is not opened")
            return False, None
        
        success, frame = self.capture.read()
        
        if not success:
            logging.warning("Failed to read frame from camera")
        
        return success, frame
    
    def stop(self):
        """Stop camera capture and release resources"""
        print("[DEBUG] === ENTERING Camera.stop() ===")
        try:
            print("[DEBUG] Camera Step 0: Checking capture object")
            if self.capture is not None:
                print("[DEBUG] Camera Step 1: Calling capture.release()")
                self.capture.release()
                print("[DEBUG] Camera Step 2: Capture released")
                self.is_opened = False
                logging.info("Camera stopped and resources released")
                print("[DEBUG] Camera Step 3: Cleanup complete")
        except Exception as e:
            print(f"[DEBUG] ERROR in Camera.stop(): {e}")
            logging.error(f"Error stopping camera: {e}")
            self.is_opened = False
        print("[DEBUG] === EXITING Camera.stop() ===")
    
    def get_properties(self):
        """Get current camera properties"""
        if not self.is_opened or self.capture is None:
            return None
        
        return {
            'width': int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.capture.get(cv2.CAP_PROP_FPS)),
            'brightness': self.capture.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.capture.get(cv2.CAP_PROP_CONTRAST),
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
