"""
Phone Detector Module (Template/Example)
Example detector showing how to create custom detectors for the proctoring system
"""

import logging
import cv2
import numpy as np
from .base_detector import BaseDetector


class PhoneDetector(BaseDetector):
    """
    Example phone detector template.
    This shows how to create a custom detector that plugs into the ProctorPipeline.
    
    To implement:
    1. Load your phone detection model in load_model()
    2. Implement detection logic in detect_phones()
    3. Draw annotations in draw_detections()
    4. Return results in process_frame()
    """
    
    def __init__(
        self,
        name="PhoneDetector",
        enabled=True,
        model_path="cv_models/phone.pt",
        confidence_threshold=0.5
    ):
        """
        Initialize phone detector
        
        Args:
            name: Name of the detector
            enabled: Whether detector is enabled
            model_path: Path to the phone detection model
            confidence_threshold: Minimum confidence for detection
        """
        super().__init__(name, enabled)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        self.logger.info(f"Initializing {name} with model: {model_path}")
    
    def load_model(self):
        """
        Load phone detection model
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            from ultralytics import YOLO
            import os
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load the YOLO model
            self.model = YOLO(self.model_path)
            
            # Verify model loaded correctly
            if self.model is None:
                self.logger.error(f"YOLO model is None after loading")
                return False
            
            self.initialized = True
            self.logger.info(f"{self.name} model loaded successfully from {self.model_path}")
            self.logger.info(f"Model info: {type(self.model)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading {self.name} model: {e}")
            self.initialized = False
            self.model = None
            return False
    
    def detect_phones(self, frame):
        """
        Detect phones in the frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            list: List of detections [(x1, y1, x2, y2, confidence), ...]
        """
        if not self.initialized or self.model is None:
            return []
        
        # Validate frame
        if frame is None or frame.size == 0:
            self.logger.warning("Invalid frame provided to phone detector")
            return []
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            # Check if results is None or empty
            if results is None:
                self.logger.warning("YOLO model returned None - model may not be properly loaded")
                return []
            
            detections = []
            
            for r in results:
                # Check if boxes exist
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                    
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error detecting phones: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes around detected phones
        
        Args:
            frame: Input frame
            detections: List of detections from detect_phones()
            
        Returns:
            Annotated frame
        """
        output_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # Draw red box for phone detection (potential cheating)
            color = (0, 0, 255)  # Red in BGR
            thickness = 3
            
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            label = f"PHONE: {confidence:.2f}"
            cv2.putText(
                output_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return output_frame
    
    def process_frame(self, frame, draw=True):
        """
        Process frame: detect phones and annotate
        
        Args:
            frame: Input frame
            draw: Whether to draw annotations
            
        Returns:
            tuple: (annotated_frame, detection_results)
        """
        if not self.enabled:
            return frame, {"enabled": False}
        
        # Detect phones
        detections = self.detect_phones(frame)
        
        # Build results dictionary
        detection_results = {
            "detector": self.name,
            "num_detections": len(detections),
            "detections": detections,
            "alert": len(detections) > 0  # Alert if phone detected
        }
        
        # Draw annotations
        if draw and len(detections) > 0:
            output_frame = self.draw_detections(frame, detections)
        else:
            output_frame = frame
        
        # Log if phone detected
        if len(detections) > 0:
            self.logger.warning(f"ALERT: {len(detections)} phone(s) detected!")
        
        return output_frame, detection_results


# You can create more detector classes following the same pattern:
# - Inherit from BaseDetector
# - Implement load_model()
# - Implement process_frame()
# - Return (annotated_frame, detection_results) dictionary
