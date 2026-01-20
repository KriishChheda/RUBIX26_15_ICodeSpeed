"""
Face Detection Module
Handles face detection using MediaPipe Tasks Vision API for optimal performance
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from .base_detector import BaseDetector

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Install with: pip install mediapipe")


class FaceDetector(BaseDetector):
    """
    Face detection using MediaPipe Tasks Vision API
    Uses Face Landmarker for detecting faces and extracting 478 facial landmarks
    (468 face mesh points + 10 iris landmarks = 478 total)
    """
    
    def __init__(
        self,
        name="FaceDetector",
        enabled=True,
        model_selection=1,  # Legacy parameter, kept for compatibility
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        """
        Initialize MediaPipe Tasks Vision face landmarker
        
        Args:
            name: Name of the detector
            enabled: Whether detector is enabled
            model_selection: Legacy parameter (not used in Tasks Vision API)
            min_detection_confidence: Minimum confidence for detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0)
        """
        super().__init__(name, enabled)
        
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")
        
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # MediaPipe Tasks Vision face landmarker
        self.face_landmarker = None
        
        # Frame counter for timestamp generation
        self.frame_count = 0
        
        self.logger.info(f"Initializing MediaPipe Tasks Vision Face Landmarker")
    
    def load_model(self, model_path=None):
        """Load MediaPipe Tasks Vision Face Landmarker model
        
        Args:
            model_path: Path to the face landmarker model file (.task)
                       If None, uses default from config
        """
        try:
            # Import config to get model path
            from .config import ProctorConfig
            
            # Use provided path or default from config
            if model_path is None:
                model_path = ProctorConfig.FACE_MARKER_MODEL_PATH
            
            # Verify model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Face landmarker model not found at: {model_path}")
            
            self.logger.info(f"Loading face landmarker model from: {model_path}")
            
            # Configure FaceLandmarker with base options
            base_options = python.BaseOptions(model_asset_path=str(model_file))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=5,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            
            self.initialized = True
            self.logger.info("MediaPipe Tasks Vision Face Landmarker loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading MediaPipe Tasks Vision Face Landmarker: {e}")
            return False
    
    def detect(self, frame):
        """
        Detect faces in frame and return face meshes using Face Landmarker
        
        Args:
            frame: Input frame (numpy array in BGR format)
            
        Returns:
            list: List of face mesh results with landmarks and bounding boxes
        """
        if not self.initialized:
            self.logger.warning("Detector not initialized")
            return []
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Generate timestamp in milliseconds for video mode
            self.frame_count += 1
            timestamp_ms = self.frame_count * 33  # Assuming ~30fps
            
            # Detect faces using Face Landmarker
            landmarker_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if not landmarker_result.face_landmarks:
                return []
            
            face_meshes = []
            
            # Process each detected face
            for face_landmarks in landmarker_result.face_landmarks:
                # Convert normalized landmarks to pixel coordinates
                landmarks = []
                x_coords = []
                y_coords = []
                
                for landmark in face_landmarks:
                    x_px = int(landmark.x * w)
                    y_px = int(landmark.y * h)
                    landmarks.append({
                        'x': x_px,
                        'y': y_px,
                        'z': landmark.z if hasattr(landmark, 'z') else 0.0
                    })
                    x_coords.append(x_px)
                    y_coords.append(y_px)
                
                # Calculate bounding box from landmarks
                if x_coords and y_coords:
                    x_min = max(0, min(x_coords))
                    y_min = max(0, min(y_coords))
                    x_max = min(w, max(x_coords))
                    y_max = min(h, max(y_coords))
                    
                    # Add padding to bounding box
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    face_data = {
                        'bbox': {
                            'x': x_min,
                            'y': y_min,
                            'w': x_max - x_min,
                            'h': y_max - y_min
                        },
                        'confidence': 1.0,  # Face Landmarker doesn't provide per-face confidence
                        'landmarks': landmarks
                    }
                    
                    face_meshes.append(face_data)
            
            return face_meshes
            
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return []
    
    def draw_faces(self, frame, face_meshes, color=(0, 255, 0), thickness=2, show_all_landmarks=False, show_landmark_numbers=False):
        """
        Draw bounding boxes and landmarks on detected faces
        
        Args:
            frame: Input frame
            face_meshes: List of face mesh data from detect()
            color: Box color (B, G, R)
            thickness: Box thickness
            show_all_landmarks: If True, shows all 478 face mesh points. If False, shows only key points
            show_landmark_numbers: If True, displays landmark index numbers (use with caution - lots of text!)
            
        Returns:
            Annotated frame
        """
        output_frame = frame.copy()
        
        for face_data in face_meshes:
            bbox = face_data['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            confidence = face_data['confidence']
            
            # Draw rectangle
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw confidence
            label = f"Face: {confidence:.2f}"
            cv2.putText(
                output_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Draw landmarks based on mode
            if face_data['landmarks'] and len(face_data['landmarks']) > 0:
                if show_all_landmarks:
                    # Draw all 478 face mesh landmarks
                    for idx, lm in enumerate(face_data['landmarks']):
                        # Use different colors for different facial regions
                        if idx < 468:  # Face contour and features
                            point_color = (0, 255, 255)  # Yellow
                        else:  # Iris landmarks (468-477)
                            point_color = (255, 0, 255)  # Magenta for iris points
                        
                        cv2.circle(output_frame, (lm['x'], lm['y']), 1, point_color, -1)
                        
                        # Optionally draw landmark numbers (warning: lots of text!)
                        if show_landmark_numbers and idx % 10 == 0:  # Show every 10th number
                            cv2.putText(output_frame, str(idx), (lm['x']+2, lm['y']-2),
                                       cv2.FONT_HERSHEY_PLAIN, 0.3, (255, 255, 255), 1)
                else:
                    # Draw only key points (eyes, nose, mouth) for clarity
                    key_indices = [33, 133, 362, 263, 1, 61, 291]  # Eyes, nose tip, mouth corners
                    for idx in key_indices:
                        if idx < len(face_data['landmarks']):
                            lm = face_data['landmarks'][idx]
                            cv2.circle(output_frame, (lm['x'], lm['y']), 2, (0, 255, 255), -1)
        
        return output_frame
    
    def process_frame(self, frame, draw=True, show_all_landmarks=False, show_landmark_numbers=False):
        """
        Process frame: detect faces
        
        Args:
            frame: Input frame
            draw: Whether to draw annotations
            show_all_landmarks: If True, shows all 478 face mesh points
            show_landmark_numbers: If True, displays landmark index numbers
            
        Returns:
            tuple: (annotated_frame, detection_results)
        """
        if not self.enabled:
            return frame, {"enabled": False}
        
        # Detect faces
        face_meshes = self.detect(frame)
        
        # Build results
        detection_results = {
            "detector": self.name,
            "num_faces": len(face_meshes),
            "face_meshes": face_meshes,
            "total_landmarks_per_face": 478  # MediaPipe Face Landmarker: 468 face + 10 iris = 478
        }
        
        # Draw annotations
        if draw and len(face_meshes) > 0:
            output_frame = self.draw_faces(frame, face_meshes, 
                                          show_all_landmarks=show_all_landmarks,
                                          show_landmark_numbers=show_landmark_numbers)
        else:
            output_frame = frame
        
        return output_frame, detection_results
    
    def cleanup(self):
        """Release MediaPipe FaceLandmarker resources"""
        print(f"[DEBUG] === ENTERING {self.name}.cleanup() ===")
        if self.face_landmarker:
            try:
                print(f"[DEBUG] {self.name} Step 0: Releasing face_landmarker")
                # MediaPipe FaceLandmarker doesn't have explicit close, set to None for GC
                self.face_landmarker = None
                print(f"[DEBUG] {self.name} Step 1: Set to None")
                self.initialized = False
                self.logger.info(f"{self.name} - MediaPipe FaceLandmarker resources released")
                print(f"[DEBUG] {self.name} Step 2: Cleanup complete")
            except Exception as e:
                print(f"[DEBUG] ERROR in {self.name}.cleanup(): {e}")
                self.logger.error(f"Error cleaning up {self.name}: {e}")
        else:
            print(f"[DEBUG] {self.name}: No resources to clean")
            self.logger.info(f"{self.name} cleanup complete (no resources loaded)")
        print(f"[DEBUG] === EXITING {self.name}.cleanup() ===")
