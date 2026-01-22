"""
Phone Detector Module
Object Detection-based phone detection using YOLO detection model
"""

import logging
import cv2
import numpy as np
from .base_detector import BaseDetector


class PhoneDetector(BaseDetector):
    """
    Phone detector using YOLO object detection model.
    Detects phones with bounding boxes, confidence scores, and precise localization.
    """
    
    def __init__(
        self,
        name="PhoneDetector",
        enabled=True,
        model_path="cv_models/phone.pt",
        confidence_threshold=0.5
    ):
        """
        Initialize phone detector (object detection model)
        
        Args:
            name: Name of the detector
            enabled: Whether detector is enabled
            model_path: Path to the phone detection model (.pt file)
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        super().__init__(name, enabled)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {0: 'phone'}  # Single class detection
        
        # Statistics tracking
        self.total_frames = 0
        self.phone_detected_frames = 0
        
        self.logger.info(f"Initializing {name} (Object Detection) with model: {model_path}")
    
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
            
            # Load the YOLO detection model
            self.model = YOLO(self.model_path)
            
            # Verify model loaded correctly
            if self.model is None:
                self.logger.error(f"YOLO model is None after loading")
                return False
            
            # Get class names if available
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                self.logger.info(f"Model classes: {self.class_names}")
            
            self.initialized = True
            self.logger.info(f"{self.name} detection model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading {self.name} model: {e}")
            self.initialized = False
            self.model = None
            return False
    
    def detect_phones(self, frame):
        """
        Detect phones in the frame with bounding boxes
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            list: List of detections, each as dict with keys:
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: Detection confidence (0-1)
                - class_id: Class ID (0 for phone)
                - class_name: Class name ('phone')
                - center: [x_center, y_center]
                - area: Bounding box area in pixels
        """
        self.total_frames += 1
        
        if not self.initialized or self.model is None:
            self.logger.debug(f"Phone detector not initialized")
            return []
        
        # Validate frame
        if frame is None or frame.size == 0:
            self.logger.warning("Invalid frame provided to phone detector")
            return []
        
        try:
            # Run YOLO object detection
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device='cpu'  # Change to 'cuda' for GPU
            )
            
            # Check if results is None or empty
            if results is None or len(results) == 0:
                return []
            
            # Get first result (single image)
            result = results[0]
            
            # Check if boxes attribute exists
            if not hasattr(result, 'boxes') or result.boxes is None:
                return []
            
            detections = []
            
            # Process each detected phone
            for box in result.boxes:
                # Extract detection information
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate additional properties
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                area = int((x2 - x1) * (y2 - y1))
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'phone'),
                    'center': [x_center, y_center],
                    'area': area
                }
                
                detections.append(detection)
            
            # Update statistics
            if len(detections) > 0:
                self.phone_detected_frames += 1
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error detecting phones: {e}", exc_info=True)
            return []
    
    def process_frame(self, frame, draw=True):
        """
        Process frame: detect phones and optionally draw bounding boxes
        
        Args:
            frame: Input frame
            draw: Whether to draw bounding boxes and labels
            
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
            "phone_detected": len(detections) > 0,
            "count": len(detections),
            "confidence": max([d['confidence'] for d in detections], default=0.0),
            "detections": detections,
            "alert": len(detections) > 0  # Alert if any phone detected
        }
        
        output_frame = frame.copy()
        
        # Draw bounding boxes if enabled
        if draw and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                # Draw bounding box (RED for alert)
                color = (0, 0, 255)  # BGR format
                thickness = 2
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with confidence
                label = f"Phone {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(
                    output_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    output_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # Draw summary at top
            summary_text = f"PHONE ALERT: {len(detections)} detected"
            cv2.rectangle(output_frame, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(
                output_frame,
                summary_text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        
        # Log if phone detected
        if len(detections) > 0:
            max_conf = max([d['confidence'] for d in detections])
            self.logger.warning(
                f"ALERT: {len(detections)} phone(s) detected "
                f"(max confidence: {max_conf:.2%})"
            )
        
        return output_frame, detection_results
    
    def get_statistics(self):
        """
        Get detection statistics
        
        Returns:
            dict: Statistics including total frames, detection rate, etc.
        """
        detection_rate = (
            (self.phone_detected_frames / self.total_frames * 100)
            if self.total_frames > 0
            else 0.0
        )
        
        return {
            'total_frames': self.total_frames,
            'phone_detected_frames': self.phone_detected_frames,
            'detection_rate': detection_rate
        }
    
    def reset_statistics(self):
        """Reset frame counters"""
        self.total_frames = 0
        self.phone_detected_frames = 0