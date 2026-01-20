"""
Eye Movement Detection Module - MediaPipe Based
Uses MediaPipe face mesh landmarks to detect and track eye movements
Analyzes pupil position and eye geometry for risk assessment
"""

import logging
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from .base_detector import BaseDetector


class EyeMovementDetector(BaseDetector):
    """Detects and tracks eye movements using MediaPipe face mesh landmarks"""
    
    def __init__(self, name="EyeMovementDetector", enabled=True):
        """
        Initialize Eye Movement Detector
        
        Args:
            name: Name of the detector
            enabled: Whether detector is enabled
        """
        super().__init__(name, enabled)
        
        # MediaPipe eye landmark indices (478-point face mesh: 468 face + 10 iris)
        # We'll use these key points for eye tracking:
        # Left Eye: outer corner (33), inner corner (133), top (159), bottom (145)
        # Right Eye: outer corner (362), inner corner (263), top (386), bottom (374)
        # Iris landmarks: Left iris center (468-472), Right iris center (473-477)
        
        self.left_eye_indices = {
            'outer': 33,
            'inner': 133,
            'top': 159,
            'bottom': 145,
            'iris_center': 468  # Left iris center landmark
        }
        
        self.right_eye_indices = {
            'outer': 362,
            'inner': 263,
            'top': 386,
            'bottom': 374,
            'iris_center': 473  # Right iris center landmark
        }
        
        # Risk analysis thresholds
        self.vertical_threshold = 0.15
        self.horizontal_min = 0.3
        self.horizontal_max = 0.7
        
        # Visualization colors
        self.keypoint_colors = {
            'outer': (255, 0, 255),    # Magenta
            'inner': (0, 255, 255),    # Yellow
            'iris': (0, 255, 0),       # Green
            'eye_contour': (255, 128, 0)  # Orange
        }
        
        # Risk status colors
        self.risk_colors = {
            'SAFE': (0, 255, 0),       # Green
            'RISK': (0, 0, 255),       # Red
            'THINKING': (255, 165, 0),  # Orange
            'CLOSED': (128, 128, 128)   # Gray
        }
        
        # Calibration state
        self.is_calibrated = False
        self.should_calibrate = False
        self.calibration_offsets = {}  # {'Left': 0.0, 'Right': 0.0}
        
        # Eye movement logging
        self.eye_movement_logger = None
        self.eye_log_file = None
        self.session_id = None
        
        self.logger.info(f"Eye Movement Detector initialized (MediaPipe-based)")
        
    def load_model(self):
        """
        Load model - No external model needed as we use MediaPipe face mesh
        
        Returns:
            bool: True (always successful)
        """
        self.initialized = True
        self.logger.info("Eye Movement Detector ready (using MediaPipe face mesh)")
        return True
    
    def setup_eye_movement_logger(self, log_dir='logs/eye_movements', session_id=None):
        """Setup dedicated eye movement logger
        
        Args:
            log_dir: Directory for eye movement logs
            session_id: Session identifier
        """
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            if session_id is None:
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.session_id = session_id
            self.eye_log_file = log_path / f"eye_movements_{session_id}.jsonl"
            
            # Create logger
            self.eye_movement_logger = logging.getLogger(f"EyeMovement_{session_id}")
            self.eye_movement_logger.setLevel(logging.INFO)
            self.eye_movement_logger.handlers.clear()
            
            # File handler for eye movement logs
            file_handler = logging.FileHandler(self.eye_log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(message)s')  # JSON lines format
            file_handler.setFormatter(file_formatter)
            self.eye_movement_logger.addHandler(file_handler)
            
            # Log header
            header = {
                'type': 'session_start',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'log_file': str(self.eye_log_file)
            }
            self.eye_movement_logger.info(json.dumps(header))
            
            self.logger.info(f"Eye movement logger initialized: {self.eye_log_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up eye movement logger: {e}")
            return False
            
    def trigger_calibration(self):
        """Trigger calibration on next frame"""
        self.should_calibrate = True
        self.logger.info("Calibration triggered for next frame")
        return True
        
    def reset_calibration(self):
        """Reset calibration data"""
        self.is_calibrated = False
        self.calibration_offsets = {}
        self.logger.info("Calibration reset")
    
    def detect(self, frame, face_meshes):
        """
        Detect eyes and extract eye data from MediaPipe face mesh
        
        Args:
            frame: Input frame (BGR)
            face_meshes: List of face mesh data from FaceDetector (must contain 'landmarks')
            
        Returns:
            list: List of eye detections with landmarks and analysis
        """
        if not self.initialized:
            self.logger.warning("Detector not initialized")
            return []
        
        if not face_meshes:
            return []
        
        h, w, _ = frame.shape
        detections = []
        
        # Process each face mesh
        for face_data in face_meshes:
            landmarks = face_data.get('landmarks')
            
            if not landmarks or len(landmarks) < 468:
                self.logger.debug("Face mesh missing landmarks (need at least 468 points)")
                continue
            
            # Process both eyes
            for eye_name, eye_indices in [('Left', self.left_eye_indices), ('Right', self.right_eye_indices)]:
                try:
                    # Extract eye landmarks
                    outer = landmarks[eye_indices['outer']]
                    inner = landmarks[eye_indices['inner']]
                    top = landmarks[eye_indices['top']]
                    bottom = landmarks[eye_indices['bottom']]
                    
                    # Get iris center if available (landmarks 468-477 for iris)
                    iris_idx = eye_indices['iris_center']
                    if iris_idx < len(landmarks):
                        iris = landmarks[iris_idx]
                    else:
                        # Fallback: estimate iris center from eye corners
                        iris = {
                            'x': int((outer['x'] + inner['x']) / 2),
                            'y': int((outer['y'] + inner['y']) / 2),
                            'z': 0.0
                        }
                    
                    # Calculate eye bounding box
                    eye_points = [outer, inner, top, bottom]
                    x_coords = [p['x'] for p in eye_points]
                    y_coords = [p['y'] for p in eye_points]
                    
                    bbox_x1 = max(0, min(x_coords) - 10)
                    bbox_y1 = max(0, min(y_coords) - 10)
                    bbox_x2 = min(w, max(x_coords) + 10)
                    bbox_y2 = min(h, max(y_coords) + 10)
                    
                    # Check if eye is open (vertical distance)
                    eye_height = abs(top['y'] - bottom['y'])
                    eye_width = abs(outer['x'] - inner['x'])
                    
                    # Eye aspect ratio for blink detection
                    ear = eye_height / (eye_width + 1e-6)
                    is_open = ear > 0.15  # Threshold for eye being open
                    
                    # Store detection
                    detection = {
                        'eye_name': eye_name,
                        'bbox': (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                        'landmarks': {
                            'outer': (outer['x'], outer['y']),
                            'inner': (inner['x'], inner['y']),
                            'top': (top['x'], top['y']),
                            'bottom': (bottom['x'], bottom['y']),
                            'iris': (iris['x'], iris['y'])
                        },
                        'eye_width': eye_width,
                        'eye_height': eye_height,
                        'eye_aspect_ratio': ear,
                        'is_open': is_open
                    }
                    
                    detections.append(detection)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing {eye_name} eye: {e}")
                    continue

        return detections 
    
    def calculate_risk(self, detection):
        """
        Calculate risk status based on iris/pupil position relative to eye
        
        Args:
            detection: Detection dictionary with eye landmarks
            
        Returns:
            tuple: (status, score, horizontal_ratio, vertical_ratio)
        """
        try:
            eye_name = detection['eye_name']
            
            # Check if eye is closed
            if not detection.get('is_open', True):
                return "EYE CLOSED", 0.0, 0.0, 0.0
            
            landmarks = detection['landmarks']
            outer = landmarks['outer']
            inner = landmarks['inner']
            top = landmarks['top']
            bottom = landmarks['bottom']
            iris = landmarks['iris']
            
            # Calculate eye dimensions
            eye_width = abs(outer[0] - inner[0]) + 1e-6  # Avoid div/0
            eye_height = abs(top[1] - bottom[1]) + 1e-6
            
            # Eye center (horizontal and vertical)
            eye_center_x = (outer[0] + inner[0]) / 2
            eye_center_y = (top[1] + bottom[1]) / 2
            
            # Iris position relative to eye center
            iris_x, iris_y = iris
            
            # Vertical ratio: how far up/down from center (normalized by width for consistency)
            raw_vertical_ratio = (iris_y - eye_center_y) / eye_width
            
            # Apply calibration offset
            calibration_offset = self.calibration_offsets.get(eye_name, 0.0)
            corrected_vertical_ratio = raw_vertical_ratio - calibration_offset
            
            # Horizontal ratio: position along the eye width
            horizontal_ratio = (iris_x - inner[0]) / eye_width
            
            # Determine status
            decision_ratio = corrected_vertical_ratio
            threshold = 0.12 if self.is_calibrated else self.vertical_threshold
            
            if decision_ratio > threshold:
                return "LOOKING DOWN (RISK)", decision_ratio, horizontal_ratio, raw_vertical_ratio
            elif decision_ratio < -threshold:
                return "LOOKING UP (THINKING)", abs(decision_ratio), horizontal_ratio, raw_vertical_ratio
            elif horizontal_ratio < self.horizontal_min or horizontal_ratio > self.horizontal_max:
                return "LOOKING SIDE (RISK)", abs(horizontal_ratio - 0.5) * 2, horizontal_ratio, raw_vertical_ratio
            else:
                return "CENTER (SAFE)", 1.0 - abs(decision_ratio), horizontal_ratio, raw_vertical_ratio
        
        except Exception as e:
            self.logger.error(f"Error in risk calculation: {e}")
            return "Error", 0.0, 0.0, 0.0
    
    def draw_eye_detection(self, frame, detection, draw_landmarks=True):
        """
        Draw eye detection with bounding box and landmarks
        
        Args:
            frame: Input frame
            detection: Detection dictionary with landmarks
            draw_landmarks: Whether to draw landmarks
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Get risk status color
        risk_status = detection.get('risk_status', 'SAFE')
        if "RISK" in risk_status:
            box_color = self.risk_colors['RISK']
        elif "THINKING" in risk_status:
            box_color = self.risk_colors['THINKING']
        elif "CLOSED" in risk_status:
            box_color = self.risk_colors['CLOSED']
        else:
            box_color = self.risk_colors['SAFE']
        
        # Draw bounding box
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw landmarks
        if draw_landmarks and detection.get('landmarks'):
            landmarks = detection['landmarks']
            
            # Draw eye corners
            outer = landmarks['outer']
            inner = landmarks['inner']
            top = landmarks['top']
            bottom = landmarks['bottom']
            iris = landmarks['iris']
            
            # Draw outer and inner corners
            cv2.circle(annotated_frame, (int(outer[0]), int(outer[1])), 3, self.keypoint_colors['outer'], -1)
            cv2.circle(annotated_frame, (int(inner[0]), int(inner[1])), 3, self.keypoint_colors['inner'], -1)
            
            # Draw top and bottom
            cv2.circle(annotated_frame, (int(top[0]), int(top[1])), 2, self.keypoint_colors['eye_contour'], -1)
            cv2.circle(annotated_frame, (int(bottom[0]), int(bottom[1])), 2, self.keypoint_colors['eye_contour'], -1)
            
            # Draw iris center (larger)
            cv2.circle(annotated_frame, (int(iris[0]), int(iris[1])), 4, self.keypoint_colors['iris'], -1)
            
            # Draw line between corners
            cv2.line(annotated_frame, (int(inner[0]), int(inner[1])), 
                    (int(outer[0]), int(outer[1])), (255, 255, 255), 1)
        
        # Draw risk status label
        if 'risk_status' in detection:
            label = detection['risk_status']
            score = detection.get('risk_score', 0.0)
            
            # Background for label
            label_text = f"{detection['eye_name']}: {label}"
            score_text = f"Score: {score:.2f}"
            
            cv2.putText(annotated_frame, label_text,
                       (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, box_color, 2)
            cv2.putText(annotated_frame, score_text,
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, box_color, 1)
        
        return annotated_frame
    
    def process_frame(self, frame, face_meshes=None, draw=True):
        """
        Process frame: detect eyes and calculate risk
        
        Args:
            frame: Input frame
            face_meshes: List of face mesh data from FaceDetector (required)
            draw: Whether to draw annotations
            
        Returns:
            tuple: (processed_frame, detection_results)
        """
        if not self.enabled:
            return frame, {"enabled": False}
        
        if face_meshes is None:
            face_meshes = []
        
        processed_frame = frame.copy()
        
        # Detect eyes using face mesh data
        detections = self.detect(frame, face_meshes)
        
        # Handle calibration flag
        if self.should_calibrate:
            self.is_calibrated = True
            self.logger.info("Calibrating eye tracking...")
        
        # Analyze risk for each detection
        risk_detections = []
        
        for detection in detections:
            # Calculate risk status
            status, score, h_ratio, raw_v_ratio = self.calculate_risk(detection)
            
            # Perform calibration if requested
            eye_name = detection['eye_name']
            if self.should_calibrate:
                self.calibration_offsets[eye_name] = raw_v_ratio
                self.logger.info(f"Calibrated {eye_name} eye with offset: {raw_v_ratio:.4f}")
                
                # Recalculate with new calibration
                status, score, h_ratio, raw_v_ratio = self.calculate_risk(detection)
            
            # Add risk analysis to detection
            detection['risk_status'] = status
            detection['risk_score'] = score
            detection['horizontal_ratio'] = h_ratio
            detection['vertical_ratio'] = raw_v_ratio
            
            risk_detections.append(detection)
            
            # Log eye movement to dedicated logger
            if self.eye_movement_logger:
                eye_log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'eye_name': eye_name,
                    'risk_status': status,
                    'risk_score': float(score),
                    'horizontal_ratio': float(h_ratio),
                    'vertical_ratio': float(raw_v_ratio),
                    'calibrated': self.is_calibrated,
                    'calibration_offset': self.calibration_offsets.get(eye_name, 0.0),
                    'eye_aspect_ratio': float(detection.get('eye_aspect_ratio', 0.0)),
                    'is_open': detection.get('is_open', True)
                }
                self.eye_movement_logger.info(json.dumps(eye_log_entry))
            
            # Draw detection
            if draw:
                processed_frame = self.draw_eye_detection(processed_frame, detection)
        
        if self.should_calibrate:
            self.should_calibrate = False
        
        # Build results
        detection_results = {
            "detector": self.name,
            "num_eyes": len(risk_detections),
            "detections": risk_detections,
            "calibrated": self.is_calibrated
        }
        
        return processed_frame, detection_results
    
    def cleanup(self):
        """Release resources and close eye movement logger"""
        try:
            # Close eye movement logger
            if self.eye_movement_logger:
                # Log session end
                end_entry = {
                    'type': 'session_end',
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat()
                }
                self.eye_movement_logger.info(json.dumps(end_entry))
                
                # Remove handlers
                for handler in self.eye_movement_logger.handlers[:]:
                    handler.close()
                    self.eye_movement_logger.removeHandler(handler)
                
                self.logger.info(f"Eye movement log saved: {self.eye_log_file}")
            
            self.initialized = False
            self.logger.info("Eye Movement Detector resources released")
        except Exception as e:
            self.logger.error(f"Error cleaning up Eye Movement Detector: {e}")
