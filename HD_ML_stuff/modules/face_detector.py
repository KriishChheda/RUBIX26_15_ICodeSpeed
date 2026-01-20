"""
Face Detection and Recognition Module
Handles face detection and verification using DeepFace library
"""

import logging
import cv2
import numpy as np
import os
from pathlib import Path
from .base_detector import BaseDetector

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Install with: pip install deepface")


class FaceDetector(BaseDetector):
    """
    Face detection and verification using DeepFace library
    Supports multiple backends and models for face recognition
    """
    
    def __init__(
        self,
        name="FaceDetector",
        enabled=True,
        detector_backend='opencv',  # 'opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe'
        model_name='Facenet',  # 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace'
        distance_metric='cosine',  # 'cosine', 'euclidean', 'euclidean_l2'
        verification_threshold=0.4,
        participant_data_path='data/participants',
        anti_spoofing=True  # Enable anti-spoofing (liveness detection)
    ):
        """
        Initialize DeepFace face detector
        
        Args:
            name: Name of the detector
            enabled: Whether detector is enabled
            detector_backend: Face detection backend to use
            model_name: Face recognition model to use
            distance_metric: Distance metric for face comparison
            verification_threshold: Threshold for face verification (lower = stricter)
            participant_data_path: Path to participant face images
            anti_spoofing: Enable anti-spoofing detection (prevents photo/video attacks)
        """
        super().__init__(name, enabled)
        
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace is required. Install with: pip install deepface")
        
        self.detector_backend = detector_backend
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.verification_threshold = verification_threshold
        self.participant_data_path = participant_data_path
        self.anti_spoofing = anti_spoofing
        
        # Single participant reference image path
        self.participant_image_path = None
        
        self.logger.info(f"Initializing DeepFace detector with backend: {detector_backend}, model: {model_name}, anti-spoofing: {anti_spoofing}")
    
    def load_model(self):
        """Load DeepFace models and participant face data"""
        try:
            # DeepFace automatically downloads and loads models on first use
            # We'll verify by running a test detection
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # This will trigger model download if not already present
            try:
                DeepFace.extract_faces(
                    test_image,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
            except Exception as e:
                self.logger.warning(f"Test detection warning (expected): {e}")
            
            # Load known participant faces
            self._load_participant_faces()
            
            self.initialized = True
            self.logger.info("DeepFace face detector loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading DeepFace: {e}")
            return False
    
    def _load_participant_faces(self):
        """Load single participant reference image"""
        # Check for participant.png or participant.jpg in data folder
        base_path = Path(self.participant_data_path)
        
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = base_path.with_name(f'participant{ext}')
            if img_path.exists():
                self.participant_image_path = str(img_path)
                self.logger.info(f"Loaded participant reference image: {img_path}")
                return
        
        self.logger.warning(f"No participant image found at data/participant.[png|jpg]. Face verification will not work.")
        self.participant_image_path = None
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using DeepFace with alignment enabled
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            list: List of detected faces with bounding boxes and facial areas (aligned)
        """
        if not self.initialized:
            self.logger.warning("Detector not initialized")
            return []
        
        try:
            face_objs = DeepFace.extract_faces(
                frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True  # Ensures face alignment for better recognition
            )
            
            return face_objs
            
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return []
    
    def verify_face(self, frame):
        """
        Verify if detected face matches the participant reference image
        Includes anti-spoofing detection if enabled
        
        Args:
            frame: Input frame containing face
            
        Returns:
            dict: Verification results containing match info and spoof detection
        """
        if not self.initialized:
            return {"verified": False, "error": "Detector not initialized"}
        
        if not self.participant_image_path:
            return {"verified": False, "error": "No participant reference image loaded"}
        
        try:
            # Step 1: Anti-spoofing check if enabled
            is_real = True
            spoof_score = None
            
            if self.anti_spoofing:
                try:
                    # DeepFace anti-spoofing using extract_faces with anti_spoofing flag
                    spoof_result = DeepFace.extract_faces(
                        frame,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        anti_spoofing=True
                    )
                    
                    if spoof_result and len(spoof_result) > 0:
                        # Check if face is real (not spoofed)
                        is_real = spoof_result[0].get('is_real', True)
                        spoof_score = spoof_result[0].get('antispoof_score', None)
                        
                        if not is_real:
                            self.logger.warning(f"Spoof detected! Score: {spoof_score}")
                            return {
                                "verified": False,
                                "is_real": False,
                                "spoof_score": spoof_score,
                                "message": "Spoof detected - photo/video attack",
                                "alert": "spoofing_attempt"
                            }
                except Exception as spoof_error:
                    self.logger.debug(f"Anti-spoofing check error: {spoof_error}")
                    # Continue with verification even if anti-spoofing fails
            
            # Step 2: Face verification with alignment
            result = DeepFace.verify(
                img1_path=frame,
                img2_path=self.participant_image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                align=True  # Ensure faces are aligned before comparison
            )
            
            if result['verified']:
                return {
                    "verified": True,
                    "distance": result['distance'],
                    "threshold": result['threshold'],
                    "model": self.model_name,
                    "is_real": is_real,
                    "spoof_score": spoof_score
                }
            else:
                return {
                    "verified": False,
                    "distance": result['distance'],
                    "threshold": result['threshold'],
                    "message": "Face does not match participant",
                    "is_real": is_real,
                    "spoof_score": spoof_score
                }
            
        except Exception as e:
            self.logger.error(f"Error verifying face: {e}")
            return {"verified": False, "error": str(e)}
    
    def draw_faces(self, frame, face_objs, verification_result=None, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected faces
        
        Args:
            frame: Input frame
            face_objs: List of face objects from detect_faces
            verification_result: Optional verification result to display
            color: Box color (B, G, R)
            thickness: Box thickness
            
        Returns:
            Annotated frame
        """
        output_frame = frame.copy()
        
        for face_obj in face_objs:
            facial_area = face_obj.get('facial_area', {})
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
            confidence = face_obj.get('confidence', 0)
            
            # Determine color based on verification
            box_color = color
            if verification_result and verification_result.get('verified'):
                box_color = (0, 255, 0)  # Green for verified
            elif verification_result is not None and not verification_result.get('verified'):
                box_color = (0, 0, 255)  # Red for not verified
            
            # Draw rectangle
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), box_color, thickness)
            
            # Draw confidence
            label = f"Face: {confidence:.2f}"
            if verification_result and verification_result.get('verified'):
                label = f"Verified: {verification_result['distance']:.2f}"
            elif verification_result and not verification_result.get('verified'):
                label = f"Not Verified: {verification_result.get('distance', 'N/A')}"
            
            cv2.putText(
                output_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )
        
        return output_frame
    
    def process_frame(self, frame, verify=False, draw=True):
        """
        Process frame: detect faces and optionally verify them
        
        Args:
            frame: Input frame
            verify: Whether to perform face verification
            draw: Whether to draw annotations
            
        Returns:
            tuple: (annotated_frame, detection_results)
        """
        if not self.enabled:
            return frame, {"enabled": False}
        
        # Detect faces
        face_objs = self.detect_faces(frame)
        
        # Verification
        verification_result = None
        if verify and len(face_objs) > 0:
            verification_result = self.verify_face(frame)
        
        # Build results
        detection_results = {
            "detector": self.name,
            "num_faces": len(face_objs),
            "faces": face_objs,
            "verification": verification_result
        }
        
        # Draw annotations
        if draw and len(face_objs) > 0:
            output_frame = self.draw_faces(frame, face_objs, verification_result)
        else:
            output_frame = frame
        
        return output_frame, detection_results
