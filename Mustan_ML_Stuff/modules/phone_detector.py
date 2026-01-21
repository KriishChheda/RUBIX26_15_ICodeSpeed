"""
Phone Detector Module
Classification-based phone detection using YOLO classification model
"""

import logging
import cv2
import numpy as np
from .base_detector import BaseDetector


class PhoneDetector(BaseDetector):
    """
    Phone detector using YOLO classification model.
    Classifies entire frame as "Phone Present" (Yes) or "No Phone" (No).
    """
    
    def __init__(
        self,
        name="PhoneDetector",
        enabled=True,
        model_path="cv_models/phone.pt",
        confidence_threshold=0.5
    ):
        """
        Initialize phone detector (classification model)
        
        Args:
            name: Name of the detector
            enabled: Whether detector is enabled
            model_path: Path to the phone classification model
            confidence_threshold: Minimum confidence for "Yes" (phone present) classification
        """
        super().__init__(name, enabled)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {}  # Will store {0: 'No', 1: 'Yes'}
        
        self.logger.info(f"Initializing {name} (Classification) with model: {model_path}")
    
    def load_model(self):
        """
        Load phone classification model
        
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
            
            # Load the YOLO classification model
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
            self.logger.info(f"{self.name} classification model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading {self.name} model: {e}")
            self.initialized = False
            self.model = None
            return False
    
    def classify_phone(self, frame):
        """
        Classify if phone is present in the frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            dict: Classification result with keys:
                - phone_detected (bool): True if phone detected
                - confidence (float): Confidence score for the prediction
                - class_id (int): Predicted class ID
                - class_name (str): Predicted class name
        """
        if not self.initialized or self.model is None:
            self.logger.info(f"Phone detector not initialized. initialized={self.initialized}, model={self.model}")
            return {
                "phone_detected": False,
                "confidence": 0.0,
                "class_id": None,
                "class_name": None,
                "error": "Model not initialized"
            }
        
        # Validate frame
        if frame is None or frame.size == 0:
            self.logger.warning("Invalid frame provided to phone detector")
            return {
                "phone_detected": False,
                "confidence": 0.0,
                "class_id": None,
                "class_name": None,
                "error": "Invalid frame"
            }
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            # Check if results is None or empty
            if results is None or len(results) == 0:
                self.logger.warning("YOLO model returned None or empty results")
                return {
                    "phone_detected": False,
                    "confidence": 0.0,
                    "class_id": None,
                    "class_name": None,
                    "error": "No results"
                }
            
            # Get first result (classification models return single result)
            result = results[0]
            
            # Access the probs (probabilities) attribute
            if not hasattr(result, 'probs') or result.probs is None:
                self.logger.warning("Result has no probs attribute")
                return {
                    "phone_detected": False,
                    "confidence": 0.0,
                    "class_id": None,
                    "class_name": None,
                    "error": "No probs in result"
                }
            
            probs = result.probs
            
            # Get top prediction
            top_class_id = int(probs.top1)  # Index of highest probability class
            top_confidence = float(probs.top1conf)  # Confidence of top prediction
            class_name = self.class_names.get(top_class_id, f"Class_{top_class_id}")
            
            # Check if phone detected (class 1 = "Yes")
            phone_detected = (top_class_id == 1 and top_confidence >= self.confidence_threshold)
            
            return {
                "phone_detected": phone_detected,
                "confidence": top_confidence,
                "class_id": top_class_id,
                "class_name": class_name,
                "all_probs": probs.data.cpu().numpy().tolist() if hasattr(probs.data, 'cpu') else probs.data
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying phone: {e}", exc_info=True)
            return {
                "phone_detected": False,
                "confidence": 0.0,
                "class_id": None,
                "class_name": None,
                "error": str(e)
            }
    
    def detect_phones(self, frame):
        """
        Legacy method - kept for compatibility
        Returns empty list since this is classification, not detection
        """
        result = self.classify_phone(frame)
        return [] if not result.get("phone_detected", False) else [(0, 0, 1, 1, result.get("confidence", 0.0))]
    
    def process_frame(self, frame, draw=True):
        """
        Process frame: classify phone presence
        
        Args:
            frame: Input frame
            draw: Whether to draw annotations (text overlay)
            
        Returns:
            tuple: (annotated_frame, classification_results)
        """
        if not self.enabled:
            return frame, {"enabled": False}
        
        # Classify phone presence
        classification = self.classify_phone(frame)
        
        # Build results dictionary
        detection_results = {
            "detector": self.name,
            "phone_detected": classification.get("phone_detected", False),
            "confidence": classification.get("confidence", 0.0),
            "class_id": classification.get("class_id"),
            "class_name": classification.get("class_name"),
            "alert": classification.get("phone_detected", False)  # Alert if phone detected
        }
        
        output_frame = frame.copy()
        
        # Draw text overlay if enabled and phone detected
        if draw and classification.get("phone_detected", False):
            class_name = classification.get("class_name", "Unknown")
            confidence = classification.get("confidence", 0.0)
            
            # Add warning overlay
            text = f"PHONE DETECTED: {class_name} ({confidence:.2%})"
            color = (0, 0, 255)  # Red in BGR
            
            # Draw background rectangle for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(output_frame, (10, 10), (20 + text_size[0], 50), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(
                output_frame,
                text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )
        
        # Log if phone detected
        if classification.get("phone_detected", False):
            self.logger.warning(f"ALERT: Phone detected with {classification.get('confidence', 0.0):.2%} confidence!")
        
        return output_frame, detection_results
