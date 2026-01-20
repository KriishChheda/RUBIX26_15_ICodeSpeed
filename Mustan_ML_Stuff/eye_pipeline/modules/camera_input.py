"""
Camera Input Module
Handles camera capture and frame retrieval with threading for performance
"""

import cv2
import logging
import threading
import time


class CameraCapture:
    """Handles camera input and frame capture with threading"""
    
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
        
        # Threading attributes
        self.grabbed = False
        self.frame = None
        self.thread = None
        self.stop_thread = False
        
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
            
            # Read first frame to ensure we have data
            self.grabbed, self.frame = self.capture.read()
            if not self.grabbed:
                logging.warning("Could not read first frame from camera")
                return False
                
            # Log actual camera properties
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            logging.info(f"Camera resolution: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Start background thread
            self.stop_thread = False
            self.thread = threading.Thread(target=self._update, args=())
            self.thread.daemon = True
            self.thread.start()
            
            logging.info("Camera started successfully (Threaded)")
            return True
            
        except Exception as e:
            logging.error(f"Error starting camera: {e}")
            return False
    
    def _update(self):
        """Thread worker function to continuously read frames"""
        while not self.stop_thread and self.is_opened:
            if self.capture.isOpened():
                grabbed, frame = self.capture.read()
                if grabbed:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    # If we can't grab a frame, pause briefly to avoid CPU spin
                    time.sleep(0.01)
            else:
                break
    
    def read_frame(self):
        """
        Retrieve the latest frame
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the image
        """
        if not self.is_opened:
            logging.warning("Camera is not opened")
            return False, None
            
        if self.grabbed and self.frame is not None:
            # Return a copy to avoid threading race conditions during processing
            return True, self.frame.copy()
        
        return False, None
    
    def stop(self):
        """Stop camera capture and release resources"""
        self.stop_thread = True
        
        # Wait for thread to finish
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            
        if self.capture is not None:
            self.capture.release()
            
        self.is_opened = False
        logging.info("Camera stopped and resources released")
    
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
