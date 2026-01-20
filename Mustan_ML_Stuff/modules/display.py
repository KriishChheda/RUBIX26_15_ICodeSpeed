"""
Display Module
Handles window creation and frame display
"""

import cv2
import logging


class DisplayWindow:
    """Handles display window and frame rendering"""
    
    def __init__(self, window_name="Camera Pipeline", fullscreen=False):
        """
        Initialize display window
        
        Args:
            window_name: Name of the display window
            fullscreen: Whether to display in fullscreen mode
        """
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.is_initialized = False
        
        logging.info(f"Initializing display window: {window_name}")
    
    def create_window(self):
        """Create the display window"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            if self.fullscreen:
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )
            
            self.is_initialized = True
            logging.info("Display window created successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error creating window: {e}")
            return False
    
    def show_frame(self, frame, fps=None):
        """
        Display a frame in the window
        
        Args:
            frame: The frame to display
            fps: Optional FPS to display on frame
            
        Returns:
            bool: True if frame was displayed successfully
        """
        if not self.is_initialized:
            logging.warning("Window not initialized")
            return False
        
        if frame is None:
            logging.warning("Received None frame")
            return False
        
        # Add FPS text if provided
        if fps is not None:
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        else:
            display_frame = frame
        
        try:
            cv2.imshow(self.window_name, display_frame)
            return True
        except Exception as e:
            logging.error(f"Error displaying frame: {e}")
            return False
    
    def check_exit_key(self, wait_time=1):
        """
        Check for exit key press (ESC or 'q')
        
        Args:
            wait_time: Time to wait for key press in milliseconds
            
        Returns:
            bool: True if exit key was pressed
        """
        key = cv2.waitKey(wait_time) & 0xFF
        return key == 27 or key == ord('q')  # ESC or 'q'
    
    def destroy(self):
        """Destroy the window and cleanup"""
        try:
            if self.is_initialized:
                cv2.destroyWindow(self.window_name)
                # Add waitKey to flush OpenCV event queue
                cv2.waitKey(1)
                self.is_initialized = False
                logging.info("Display window destroyed")
        except Exception as e:
            logging.error(f"Error destroying window: {e}")
            self.is_initialized = False
    
    def destroy_all(self):
        """Destroy all OpenCV windows"""
        print("[DEBUG] === ENTERING Display.destroy_all() ===")
        try:
            print("[DEBUG] Display Step 0: Calling cv2.destroyAllWindows()")
            cv2.destroyAllWindows()
            print("[DEBUG] Display Step 1: destroyAllWindows() returned")
            # Add waitKey to flush OpenCV event queue and prevent hanging
            print("[DEBUG] Display Step 2: Calling cv2.waitKey(1)")
            cv2.waitKey(1)
            print("[DEBUG] Display Step 3: waitKey(1) returned")
            self.is_initialized = False
            logging.info("All display windows destroyed")
            print("[DEBUG] Display Step 4: Cleanup complete")
        except Exception as e:
            print(f"[DEBUG] ERROR in Display.destroy_all(): {e}")
            logging.error(f"Error destroying windows: {e}")
            self.is_initialized = False
        print("[DEBUG] === EXITING Display.destroy_all() ===")
    
    def set_position(self, x, y):
        """Set window position on screen"""
        if self.is_initialized:
            cv2.moveWindow(self.window_name, x, y)
    
    def resize_window(self, width, height):
        """Resize the window"""
        if self.is_initialized:
            cv2.resizeWindow(self.window_name, width, height)
    
    def __enter__(self):
        """Context manager entry"""
        self.create_window()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.destroy_all()
