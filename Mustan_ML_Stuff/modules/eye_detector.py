"""
Eye Movement Detection Module - MediaPipe Based
Uses MediaPipe Face Landmarker with 478 landmarks for superior eye and iris tracking
Analyzes iris position and eye geometry for gaze direction detection
"""

import logging
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from .base_detector import BaseDetector

# MediaPipe eye and iris landmark indices (478-point face mesh)
# These indices match the FaceDetector's MediaPipe Face Mesh output
LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 380, 374, 373]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


class EyeMovementDetector(BaseDetector):
    """Detects and tracks eye movements using MediaPipe Face Landmarker (Tasks API)"""
    
    def __init__(self, name="EyeMovementDetector", enabled=True, config=None):
        """
        Initialize Eye Movement Detector
        
        Args:
            name: Name of the detector
            enabled: Whether detector is enabled
            config: Configuration object with eye tracking settings
        """
        super().__init__(name, enabled)
        
        # Configuration
        self.config = config
        
        # Detection thresholds from config
        if config:
            # Detection thresholds from config
            self.horizontal_threshold = getattr(config, 'EYE_HORIZONTAL_THRESHOLD', 0.30)
            self.vertical_down_threshold = getattr(config, 'EYE_VERTICAL_DOWN_THRESHOLD', 0.10)
            self.vertical_up_threshold = getattr(config, 'EYE_VERTICAL_UP_THRESHOLD', -0.30)
            self.closed_threshold = getattr(config, 'EYE_CLOSED_THRESHOLD', 0.15)
            
            # MediaPipe confidence settings
            self.face_detection_confidence = getattr(config, 'EYE_FACE_DETECTION_CONFIDENCE', 0.5)
            self.face_presence_confidence = getattr(config, 'EYE_FACE_PRESENCE_CONFIDENCE', 0.5)
            self.tracking_confidence = getattr(config, 'EYE_TRACKING_CONFIDENCE', 0.5)
            
            # Visualization settings
            self.draw_landmarks = getattr(config, 'EYE_DRAW_LANDMARKS', True)
            self.debug_mode = getattr(config, 'EYE_DEBUG_MODE', False)
            
            # Alert settings
            self.enable_looking_down_alert = getattr(config, 'EYE_ENABLE_LOOKING_DOWN_ALERT', True)
            self.enable_looking_away_alert = getattr(config, 'EYE_ENABLE_LOOKING_AWAY_ALERT', True)
            self.enable_no_face_alert = getattr(config, 'EYE_ENABLE_NO_FACE_ALERT', True)
        else:
            # Default values
            self.model_path = 'cv_models/face_landmarker.task'
            self.model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            self.horizontal_threshold = 0.30
            self.vertical_down_threshold = 0.10
            self.vertical_up_threshold = -0.30
            self.closed_threshold = 0.15
            self.face_detection_confidence = 0.5
            self.face_presence_confidence = 0.5
            self.tracking_confidence = 0.5
            self.draw_landmarks = True
            self.debug_mode = False
            self.enable_looking_down_alert = True
            self.enable_looking_away_alert = True
            self.enable_no_face_alert = True
        
        # Visualization colors
        self.keypoint_colors = {
            'eye_bbox': (0, 255, 0),     # Green for safe
            'eye_bbox_risk': (0, 0, 255), # Red for risk
            'eye_center': (255, 255, 0),  # Yellow
            'iris_center': (0, 255, 255), # Cyan
            'iris_points': (255, 0, 255)  # Magenta
        }
        
        # Eye movement logging
        self.eye_movement_logger = None
        self.eye_log_file = None
        self.session_id = None
        
        self.logger.info(f"Eye Movement Detector initialized (MediaPipe Face Landmarker)")
        
    
    def load_model(self):
        """
        Initialize eye detector (no separate model needed, uses FaceDetector's face mesh)
        
        Returns:
            bool: True (always successful)
        """
        self.initialized = True
        self.logger.info("Eye Movement Detector initialized (uses shared FaceDetector face mesh)")
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
    
    def _get_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        
        Args:
            eye_landmarks: Array of eye landmark coordinates
            
        Returns:
            float: Eye aspect ratio
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def _get_gaze_direction(self, eye_center, iris_center, eye_width, eye_height):
        """
        Determine gaze direction based on iris position relative to eye center
        
        Args:
            eye_center: Eye center coordinates (x, y)
            iris_center: Iris center coordinates (x, y)
            eye_width: Eye width in pixels
            eye_height: Eye height in pixels
            
        Returns:
            tuple: (direction_status, is_risky, horizontal_ratio, vertical_ratio)
        """
        # Normalize by eye dimensions
        horizontal_ratio = (iris_center[0] - eye_center[0]) / (eye_width / 2 + 1e-6)
        vertical_ratio = (iris_center[1] - eye_center[1]) / (eye_height / 2 + 1e-6)
        
        # Determine direction with priority to vertical
        if vertical_ratio > self.vertical_down_threshold:
            return "LOOKING DOWN", True, horizontal_ratio, vertical_ratio
        elif vertical_ratio < self.vertical_up_threshold:
            return "LOOKING UP", False, horizontal_ratio, vertical_ratio
        elif horizontal_ratio > self.horizontal_threshold:
            return "LOOKING RIGHT", True, horizontal_ratio, vertical_ratio
        elif horizontal_ratio < -self.horizontal_threshold:
            return "LOOKING LEFT", True, horizontal_ratio, vertical_ratio
        else:
            return "CENTER", False, horizontal_ratio, vertical_ratio
    
    def _process_eye(self, landmarks, eye_indices, iris_indices, img_w, img_h, eye_name):
        """
        Process eye landmarks to extract eye center, iris center, and dimensions
        
        Args:
            landmarks: List of landmark dicts from FaceDetector with format {'x': int, 'y': int, 'z': float}
            eye_indices: List of landmark indices for eye contour
            iris_indices: List of landmark indices for iris
            img_w: Image width
            img_h: Image height
            eye_name: Name of the eye (Left/Right)
            
        Returns:
            dict: Eye data including center, iris position, dimensions, and EAR
        """
        # Get eye landmark coordinates (already in pixel coordinates from FaceDetector)
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                point = landmarks[idx]
                x = point['x']
                y = point['y']
                eye_points.append([x, y])
        
        eye_points = np.array(eye_points)
        
        # Calculate eye center and dimensions
        eye_center = eye_points.mean(axis=0).astype(int)
        eye_left = eye_points[:, 0].min()
        eye_right = eye_points[:, 0].max()
        eye_top = eye_points[:, 1].min()
        eye_bottom = eye_points[:, 1].max()
        
        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top
        
        # Calculate Eye Aspect Ratio for blink detection
        ear = self._get_eye_aspect_ratio(eye_points)
        
        # Get iris center (landmarks 468-477 are iris points)
        iris_points = []
        for idx in iris_indices:
            if idx < len(landmarks):
                point = landmarks[idx]
                x = point['x']
                y = point['y']
                iris_points.append([x, y])
        
        iris_center = np.array(iris_points).mean(axis=0).astype(int)
        
        return {
            'eye_name': eye_name,
            'eye_center': eye_center,
            'iris_center': iris_center,
            'eye_width': eye_width,
            'eye_height': eye_height,
            'eye_points': eye_points,
            'iris_points': iris_points,
            'ear': ear,
            'bbox': (eye_left, eye_top, eye_right, eye_bottom)
        }
    
    def detect(self, frame, face_meshes):
        """
        Detect eyes and analyze gaze direction from pre-detected face meshes
        
        Args:
            frame: Input frame (BGR)
            face_meshes: List of face mesh data from FaceDetector (must contain 'landmarks')
            
        Returns:
            list: List of eye detections with gaze analysis
        """
        if not self.initialized:
            self.logger.warning("Detector not initialized")
            return []
        
        if not face_meshes:
            if self.enable_no_face_alert:
                return [{
                    'eye_name': 'None',
                    'status': 'NO FACE',
                    'is_risky': True,
                    'alert': True
                }]
            return []
        
        h, w, _ = frame.shape
        all_detections = []
        
        # Process each face mesh
        for face_data in face_meshes:
            landmarks = face_data.get('landmarks')
            
            if not landmarks or len(landmarks) < 478:
                self.logger.debug("Face mesh missing landmarks (need at least 478 points for iris tracking)")
                continue
            
            try:
                # Process both eyes
                left_eye_data = self._process_eye(
                    landmarks, 
                    LEFT_EYE, 
                    LEFT_IRIS, 
                    w, h,
                    'Left'
                )
                
                right_eye_data = self._process_eye(
                    landmarks, 
                    RIGHT_EYE, 
                    RIGHT_IRIS, 
                    w, h,
                    'Right'
                )
                
                # Analyze each eye
                for eye_data in [left_eye_data, right_eye_data]:
                    # Check if eye is closed
                    if eye_data['ear'] < self.closed_threshold:
                        eye_data['status'] = "EYES CLOSED"
                        eye_data['is_risky'] = False
                        eye_data['horizontal_ratio'] = 0.0
                        eye_data['vertical_ratio'] = 0.0
                        eye_data['alert'] = False
                    else:
                        # Determine gaze direction
                        status, is_risky, h_ratio, v_ratio = self._get_gaze_direction(
                            eye_data['eye_center'],
                            eye_data['iris_center'],
                            eye_data['eye_width'],
                            eye_data['eye_height']
                        )
                        
                        eye_data['status'] = status
                        eye_data['is_risky'] = is_risky
                        eye_data['horizontal_ratio'] = h_ratio
                        eye_data['vertical_ratio'] = v_ratio
                        
                        # Determine if alert should be raised
                        alert = False
                        if "LOOKING DOWN" in status and self.enable_looking_down_alert:
                            alert = True
                        elif ("LOOKING LEFT" in status or "LOOKING RIGHT" in status) and self.enable_looking_away_alert:
                            alert = True
                        
                        eye_data['alert'] = alert
                    
                    all_detections.append(eye_data)
            
            except Exception as e:
                self.logger.error(f"Error processing eyes: {e}")
                continue
        
        return all_detections 
    
    
    def calculate_risk(self, detection):
        """
        Calculate risk status from detection data (for compatibility with proctor_pipeline)
        
        Args:
            detection: Detection dictionary with eye data
            
        Returns:
            tuple: (status, score, horizontal_ratio, vertical_ratio)
        """
        status = detection.get('status', 'UNKNOWN')
        is_risky = detection.get('is_risky', False)
        h_ratio = detection.get('horizontal_ratio', 0.0)
        v_ratio = detection.get('vertical_ratio', 0.0)
        
        # Calculate score (0-1, higher = more suspicious)
        if is_risky:
            score = max(abs(h_ratio), abs(v_ratio))
        else:
            score = 0.0
        
        return status, score, h_ratio, v_ratio
    
    def _draw_eye_tracking(self, frame, eye_data):
        """
        Draw eye tracking visualization on frame
        
        Args:
            frame: Input frame
            eye_data: Eye detection data
            
        Returns:
            Annotated frame
        """
        is_risky = eye_data.get('is_risky', False)
        
        # Draw eye bounding box
        x1, y1, x2, y2 = eye_data['bbox']
        color = self.keypoint_colors['eye_bbox_risk'] if is_risky else self.keypoint_colors['eye_bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw eye center
        cv2.circle(frame, tuple(eye_data['eye_center']), 3, self.keypoint_colors['eye_center'], -1)
        
        # Draw iris center
        cv2.circle(frame, tuple(eye_data['iris_center']), 5, self.keypoint_colors['iris_center'], -1)
        
        # Draw iris points
        for point in eye_data['iris_points']:
            cv2.circle(frame, tuple(point), 2, self.keypoint_colors['iris_points'], -1)
        
        # Draw status label
        status = eye_data.get('status', 'UNKNOWN')
        label = f"{eye_data['eye_name']}: {status}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Debug mode - show ratios
        if self.debug_mode:
            h_ratio = eye_data.get('horizontal_ratio', 0.0)
            v_ratio = eye_data.get('vertical_ratio', 0.0)
            ear = eye_data.get('ear', 0.0)
            
            cv2.putText(frame, f"H:{h_ratio:.2f} V:{v_ratio:.2f}", (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"EAR:{ear:.2f}", (x1, y2 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame
        
        return frame
    
    def process_frame(self, frame, face_meshes=None, draw=True):
        """
        Process frame: detect eyes and calculate gaze direction
        
        Args:
            frame: Input frame (BGR)
            face_meshes: Not used (kept for compatibility)
            draw: Whether to draw annotations
            
        Returns:
            tuple: (processed_frame, detection_results)
        """
        if not self.enabled:
            return frame, {"enabled": False}
        
        processed_frame = frame.copy()
        
        # Detect eyes and analyze gaze
        detections = self.detect(frame, face_meshes)
        
        # Draw visualizations
        if draw and self.draw_landmarks:
            for detection in detections:
                if detection.get('eye_name') != 'None':  # Skip "NO FACE" detection
                    processed_frame = self._draw_eye_tracking(processed_frame, detection)
        
        # Log eye movements if logger is configured
        if self.eye_movement_logger:
            for detection in detections:
                if detection.get('eye_name') != 'None':
                    eye_log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'eye_name': detection.get('eye_name'),
                        'status': detection.get('status'),
                        'is_risky': detection.get('is_risky'),
                        'horizontal_ratio': float(detection.get('horizontal_ratio', 0.0)),
                        'vertical_ratio': float(detection.get('vertical_ratio', 0.0)),
                        'ear': float(detection.get('ear', 0.0)),
                        'alert': detection.get('alert', False)
                    }
                    self.eye_movement_logger.info(json.dumps(eye_log_entry))
        
        # Build results
        detection_results = {
            "detector": self.name,
            "num_eyes": len([d for d in detections if d.get('eye_name') != 'None']),
            "detections": detections,
            "alert": any(d.get('alert', False) for d in detections)
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
                
                if self.eye_log_file:
                    self.logger.info(f"Eye movement log saved: {self.eye_log_file}")
            
            self.initialized = False
            self.logger.info("Eye Movement Detector resources released")
        except Exception as e:
            self.logger.error(f"Error cleaning up Eye Movement Detector: {e}")
