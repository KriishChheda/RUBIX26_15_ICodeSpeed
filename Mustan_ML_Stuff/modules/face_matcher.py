"""
Face Matcher Module
Handles face verification/matching using DeepFace with configurable backends
Optimized to accept pre-cropped faces to skip redundant detection
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from .base_detector import BaseDetector

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Install with: pip install deepface")


class FaceMatcher(BaseDetector):
    """
    Face matching/verification using DeepFace with configurable backends
    Optimized to accept pre-cropped faces and skip redundant detection
    Caches participant embedding for fast real-time verification
    Returns boolean match results
    """
    
    def __init__(
        self,
        name="FaceMatcher",
        enabled=True,
        model_name='Facenet512',  # DeepFace model: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace, Dlib, SFace
        distance_metric='cosine',  # Distance metric: cosine, euclidean, euclidean_l2
        distance_threshold=0.4,  # Distance threshold for matching (lower = stricter, varies by model)
        participant_image_path='data/participant.png'
    ):
        """
        Initialize DeepFace face matcher with configurable backend
        
        Args:
            name: Name of the matcher
            enabled: Whether matcher is enabled
            model_name: DeepFace model backend (VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace, Dlib, SFace)
            distance_metric: Distance metric for comparison (cosine, euclidean, euclidean_l2)
            distance_threshold: Distance threshold for matching (model-specific, lower = stricter)
            participant_image_path: Path to participant reference image
            
        Note:
            Recommended thresholds by model (cosine distance):
            - VGG-Face: 0.40
            - Facenet: 0.40
            - Facenet512: 0.30
            - ArcFace: 0.68
            - Dlib: 0.07
            - SFace: 0.593
            - OpenFace: 0.10
        """
        super().__init__(name, enabled)
        
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace is required. Install with: pip install deepface")
        
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.distance_threshold = distance_threshold
        self.participant_image_path = participant_image_path
        
        # Cached participant embedding (computed once, reused for all frames)
        self.participant_embedding = None
        
        # Cached embedding dimensions
        self.embedding_dim = None
        
        self.logger.info(f"Initializing DeepFace matcher with model={model_name}, metric={distance_metric}, threshold={distance_threshold}")
    
    def load_model(self):
        """Load DeepFace model and compute participant embedding"""
        try:
            # DeepFace models are loaded lazily on first use
            # We don't need to pre-load the model - it will load automatically when needed
            self.logger.info(f"Initializing DeepFace matcher with model={self.model_name}")
            self.logger.info("Model will be loaded automatically on first use")
            
            # Load and cache participant embedding
            # This will trigger the model loading when it extracts the embedding
            self._load_participant_embedding()
            
            # Check if participant embedding was successfully loaded
            if self.participant_embedding is not None:
                self.initialized = True
                self.logger.info("DeepFace face matcher loaded successfully")
                return True
            else:
                self.logger.error("Failed to load participant embedding")
                self.initialized = False
                return False
            
        except Exception as e:
            self.logger.error(f"Error loading DeepFace: {e}", exc_info=True)
            self.initialized = False
            return False
    
    def _load_participant_embedding(self):
        """Load participant image and compute embedding for caching"""
        try:
            # Check for participant image
            participant_path = Path(self.participant_image_path)
            
            # Try different extensions if exact path doesn't exist
            if not participant_path.exists():
                base_path = participant_path.parent / participant_path.stem
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = base_path.with_suffix(ext)
                    if test_path.exists():
                        participant_path = test_path
                        break
            
            if not participant_path.exists():
                self.logger.warning(f"Participant image not found at {self.participant_image_path}")
                self.participant_embedding = None
                return
            
            self.logger.info(f"Loading participant image: {participant_path}")
            
            # Load and preprocess image
            participant_image = cv2.imread(str(participant_path))
            if participant_image is None:
                self.logger.error("Failed to load participant image")
                self.participant_embedding = None
                return
            
            self.logger.info(f"Participant image loaded: {participant_image.shape}")
            
            # Extract embedding - USE DETECTION for participant image since it's likely a full photo
            # We use enforce_detection=True here to detect the face first
            self.participant_embedding = self._extract_embedding_with_detection(participant_image)
            
            if self.participant_embedding is not None:
                self.logger.info(f"Participant embedding cached: shape={self.participant_embedding.shape}")
            else:
                self.logger.error("Failed to extract participant embedding - ensure participant.png contains a clear face")
                
        except Exception as e:
            self.logger.error(f"Error loading participant embedding: {e}")
            self.participant_embedding = None
    
    def _extract_embedding_with_detection(self, full_image):
        """
        Extract embedding from a full image by detecting face first
        Used for loading participant image which may not be pre-cropped
        
        Args:
            full_image: Full image that may contain a face (numpy array BGR)
            
        Returns:
            numpy array: Face embedding vector
        """
        try:
            if full_image is None or full_image.size == 0:
                return None
            
            self.logger.info("Extracting embedding with face detection enabled...")
            
            # Extract embedding using DeepFace with face detection enabled
            embedding_objs = DeepFace.represent(
                img_path=full_image,
                model_name=self.model_name,
                enforce_detection=True,  # Enable detection for full images
                detector_backend='opencv',  # Use opencv for detection
                align=True,
                normalization='base'
            )
            
            if not embedding_objs or len(embedding_objs) == 0:
                self.logger.error("No face detected in participant image")
                return None
            
            # Get embedding vector from first detected face
            embedding = np.array(embedding_objs[0]['embedding'])
            
            # Store embedding dimension
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
                self.logger.info(f"Embedding dimension: {self.embedding_dim}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding with detection: {e}")
            return None
    
    def _extract_embedding(self, face_roi):
        """
        Extract embedding from face region using DeepFace
        Uses enforce_detection=False to skip redundant face detection
        
        Args:
            face_roi: Pre-cropped face region image (numpy array BGR)
            
        Returns:
            numpy array: Face embedding vector (dimensionality varies by model)
        """
        try:
            if face_roi is None or face_roi.size == 0:
                return None
            
            # Ensure minimum size for embedding extraction
            if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                self.logger.debug("Face ROI too small for embedding extraction")
                return None
            
            # Extract embedding using DeepFace.represent()
            # enforce_detection=False allows us to skip face detection since face is already cropped
            # This is the key optimization - we pass pre-cropped faces directly to embedding extraction
            embedding_objs = DeepFace.represent(
                img_path=face_roi,
                model_name=self.model_name,
                enforce_detection=False,  # Skip detection, face is already cropped
                detector_backend='skip',  # Skip detection backend entirely
                align=True,  # Still align the face for better accuracy
                normalization='base'  # Apply normalization
            )
            
            if not embedding_objs or len(embedding_objs) == 0:
                self.logger.debug("No embedding extracted")
                return None
            
            # Get embedding vector
            embedding = np.array(embedding_objs[0]['embedding'])
            
            # Store embedding dimension on first extraction
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
                self.logger.info(f"Embedding dimension: {self.embedding_dim}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding: {e}")
            return None
    
    def _compute_distance(self, embedding1, embedding2):
        """
        Compute distance between two embeddings based on configured metric
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            float: Distance score (lower = more similar)
        """
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            # cosine_similarity = dot(A, B) / (||A|| * ||B||)
            distance = 1 - np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif self.distance_metric == 'euclidean':
            # Euclidean distance
            distance = np.linalg.norm(embedding1 - embedding2)
        elif self.distance_metric == 'euclidean_l2':
            # Euclidean L2 normalized
            distance = np.linalg.norm(
                embedding1 / np.linalg.norm(embedding1) - 
                embedding2 / np.linalg.norm(embedding2)
            )
        else:
            # Default to cosine
            distance = 1 - np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        
        return float(distance)
    
    def match(self, face_roi):
        """
        Match face against cached participant embedding
        
        Args:
            face_roi: Pre-cropped face region to match (numpy array BGR)
            
        Returns:
            bool: True if face matches participant, False otherwise
        """
        if not self.initialized:
            self.logger.warning("Matcher not initialized")
            return False
        
        if self.participant_embedding is None:
            self.logger.warning("No participant embedding cached")
            return False
        
        try:
            # Extract embedding from current face (no detection, already cropped)
            current_embedding = self._extract_embedding(face_roi)
            
            if current_embedding is None:
                return False
            
            # Compute distance (lower = more similar)
            distance = self._compute_distance(current_embedding, self.participant_embedding)
            
            # Check if match (distance below threshold)
            is_match = distance < self.distance_threshold
            
            self.logger.debug(f"Match: {is_match}, Distance: {distance:.4f}, Threshold: {self.distance_threshold}")
            
            return is_match
            
        except Exception as e:
            self.logger.error(f"Error matching face: {e}")
            return False
    
    def match_with_details(self, face_roi):
        """
        Match face and return detailed results
        
        Args:
            face_roi: Pre-cropped face region to match (numpy array BGR)
            
        Returns:
            dict: Match results with distance and confidence
        """
        if not self.initialized:
            return {
                "matched": False,
                "error": "Matcher not initialized"
            }
        
        if self.participant_embedding is None:
            return {
                "matched": False,
                "error": "No participant embedding cached"
            }
        
        try:
            # Extract embedding (no detection, face is already cropped)
            current_embedding = self._extract_embedding(face_roi)
            
            if current_embedding is None:
                return {
                    "matched": False,
                    "error": "Failed to extract embedding"
                }
            
            # Compute distance (lower = more similar)
            distance = self._compute_distance(current_embedding, self.participant_embedding)
            
            # Compute confidence (inverse of distance, normalized by threshold)
            # Confidence is higher when distance is lower
            confidence = max(0.0, min(1.0, 1.0 - (distance / self.distance_threshold)))
            
            # Check if match (distance below threshold)
            is_match = distance < self.distance_threshold
            
            return {
                "matched": is_match,
                "distance": float(distance),
                "confidence": float(confidence),
                "threshold": self.distance_threshold,
                "metric": self.distance_metric,
                "model": self.model_name,
                "message": "Match found" if is_match else "No match"
            }
            
        except Exception as e:
            self.logger.error(f"Error matching face: {e}")
            return {
                "matched": False,
                "error": str(e)
            }
    
    def process_frame(self, frame, face_meshes):
        """
        Process frame with detected faces and return match results
        
        Args:
            frame: Input frame
            face_meshes: List of face mesh data from FaceDetector
            
        Returns:
            tuple: (frame, match_results)
        """
        if not self.enabled:
            return frame, {"enabled": False}
        
        match_results = []
        
        for face_data in face_meshes:
            bbox = face_data['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Match face
            match_result = self.match_with_details(face_roi)
            match_result['bbox'] = bbox
            
            match_results.append(match_result)
        
        results = {
            "matcher": self.name,
            "num_faces": len(face_meshes),
            "matches": match_results
        }
        
        return frame, results
    
    def cleanup(self):
        """Release DeepFace model and cached embeddings"""
        print(f"[DEBUG] === ENTERING {self.name}.cleanup() ===")
        try:
            print(f"[DEBUG] {self.name} Step 0: Clearing embeddings")
            # Clear cached embeddings
            self.participant_embedding = None
            print(f"[DEBUG] {self.name} Step 1: participant_embedding = None")
            self.embedding_dim = None
            print(f"[DEBUG] {self.name} Step 2: embedding_dim = None")
            self.initialized = False
            print(f"[DEBUG] {self.name} Step 3: initialized = False")
            
            # DeepFace models are managed by the library internally
            # Setting to None allows garbage collection
            self.logger.info(f"{self.name} - DeepFace model and embeddings released")
            print(f"[DEBUG] {self.name} Step 4: Cleanup complete")
        except Exception as e:
            print(f"[DEBUG] ERROR in {self.name}.cleanup(): {e}")
            self.logger.error(f"Error cleaning up {self.name}: {e}")
        print(f"[DEBUG] === EXITING {self.name}.cleanup() ===")
