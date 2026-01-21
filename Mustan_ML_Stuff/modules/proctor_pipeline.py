"""
Proctor Module
Main proctoring pipeline that orchestrates multiple detection modules
"""

import logging
import cv2
import time
from .camera_pipeline import CameraPipeline
from .base_detector import BaseDetector
from .proctor_logger import ProctorLogger
from .alert_communicator import AlertCommunicator
from .face_detector import FaceDetector
from .face_matcher import FaceMatcher
from .eye_detector import EyeMovementDetector
from .phone_detector import PhoneDetector


class ProctorPipeline(CameraPipeline):
    """
    Proctoring pipeline that inherits from CameraPipeline.
    Automatically loads and manages detector modules based on configuration.
    """
    
    def __init__(self, config=None, frame_skip=2, session_id=None):
        """
        Initialize proctoring pipeline and auto-load detectors based on config
        
        Args:
            config: Configuration object (ProctorConfig)
            frame_skip: Number of frames to skip between processing (default: 2, process every 3rd frame)
            session_id: Optional session ID for logging
        """
        super().__init__(config)
        
        # Detector management
        self.detectors = {}
        self.face_detector = None
        self.face_matcher = None
        self.eye_detector = None
        self.phone_detector = None
        self.frame_skip = frame_skip
        self.frame_counter = 0
        
        # Session logger
        self.session_logger = ProctorLogger(
            log_dir=config.PROCTORING_LOG_DIR,
            session_id=session_id
        )
        
        # Alert communicator for real-time frontend notifications
        self.alert_comm = AlertCommunicator(
            log_dir=config.PROCTORING_LOG_DIR,
            alert_file_name=config.ALERT_STATE_FILE_NAME,
            write_interval=0.1,  # Write every 100ms max
            cooldown_duration=1.0  # 1 second cooldown per alert type
        )
        
        # Proctoring state
        self.proctoring_results = {
            "total_frames_captured": 0,
            "total_frames_processed": 0,
            "detections": {},
            "alerts": []
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Proctoring pipeline initialized with frame_skip={frame_skip}")
        
        # Auto-load detectors based on configuration
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize all detectors based on configuration settings"""
        self.logger.info("Auto-loading detectors based on configuration...")
        
        # 1. Face Detector (MediaPipe) - Required for most features
        if getattr(self.config, 'FACE_DETECT_ENABLE', True):
            try:
                self.logger.info("Loading Face Detector (MediaPipe)...")
                self.face_detector = FaceDetector(
                    name="FaceDetector",
                    enabled=True,
                    model_selection=getattr(self.config, 'FACE_MODEL_SELECTION', 1),
                    min_detection_confidence=getattr(self.config, 'FACE_MIN_DETECTION_CONFIDENCE', 0.7),
                    min_tracking_confidence=getattr(self.config, 'FACE_MIN_TRACKING_CONFIDENCE', 0.5)
                )
                
                if self.face_detector.load_model(getattr(self.config, 'FACE_MARKER_MODEL_PATH', None)):
                    self.detectors['FaceDetector'] = self.face_detector
                    self.proctoring_results["detections"]['FaceDetector'] = []
                    self.logger.info("✓ Face Detector loaded successfully")
                else:
                    self.logger.error("✗ Failed to load Face Detector")
                    self.face_detector = None
            except Exception as e:
                self.logger.error(f"Error loading Face Detector: {e}")
                self.face_detector = None
        
        # 2. Face Matcher (DeepFace) - For identity verification
        if getattr(self.config, 'FACE_MATCH_ENABLE', False):
            try:
                self.logger.info("Loading Face Matcher (DeepFace)...")
                self.face_matcher = FaceMatcher(
                    name="FaceMatcher",
                    enabled=True,
                    model_name=getattr(self.config, 'FACE_MATCHING_BACKEND', 'Facenet'),
                    distance_metric=getattr(self.config, 'FACE_MATCHING_DISTANCE_METRIC', 'cosine'),
                    distance_threshold=getattr(self.config, 'FACE_MATCHING_THRESHOLD', 0.5),
                    participant_image_path=getattr(self.config, 'PARTICIPANT_DATA_PATH', 'data/participant.png')
                )
                
                if self.face_matcher.load_model():
                    self.detectors['FaceMatcher'] = self.face_matcher
                    self.proctoring_results["detections"]['FaceMatcher'] = []
                    self.logger.info("✓ Face Matcher loaded successfully")
                else:
                    self.logger.error("✗ Failed to load Face Matcher")
                    self.face_matcher = None
            except Exception as e:
                self.logger.error(f"Error loading Face Matcher: {e}")
                self.face_matcher = None
        
        # 3. Eye Movement Detector (MediaPipe-based)
        if getattr(self.config, 'EYE_TRACKING_ENABLE', False):
            try:
                self.logger.info("Loading Eye Movement Detector (MediaPipe-based)...")
                self.eye_detector = EyeMovementDetector(
                    name="EyeMovementDetector",
                    enabled=True
                )
                
                if self.eye_detector.load_model():
                    self.detectors['EyeMovementDetector'] = self.eye_detector
                    self.proctoring_results["detections"]['EyeMovementDetector'] = []
                    
                    # Setup eye movement logger
                    if hasattr(self.eye_detector, 'setup_eye_movement_logger'):
                        log_dir = getattr(self.config, 'EYE_MOVEMENT_LOG_DIR', 'logs/eye_movements')
                        self.eye_detector.setup_eye_movement_logger(
                            log_dir=log_dir,
                            session_id=self.session_logger.session_id
                        )
                    
                    self.logger.info("✓ Eye Movement Detector loaded successfully")
                else:
                    self.logger.error("✗ Failed to load Eye Movement Detector")
                    self.eye_detector = None
            except Exception as e:
                self.logger.error(f"Error loading Eye Movement Detector: {e}")
                self.eye_detector = None
        
        # 4. Phone Detector (if enabled and available)
        if getattr(self.config, 'PHONE_DETECT_ENABLE', False):
            try:
                self.logger.info("Loading Phone Detector...")
                self.phone_detector = PhoneDetector(
                    name="PhoneDetector",
                    enabled=True,
                    model_path=getattr(self.config, 'PHONE_MODEL_PATH', 'cv_models/phone.pt'),
                    confidence_threshold=getattr(self.config, 'PHONE_CONFIDENCE_THRESHOLD', 0.5)
                )
                
                if self.phone_detector.load_model():
                    self.detectors['PhoneDetector'] = self.phone_detector
                    self.proctoring_results["detections"]['PhoneDetector'] = []
                    self.logger.info("✓ Phone Detector loaded successfully")
                else:
                    self.logger.error("✗ Failed to load Phone Detector")
                    self.phone_detector = None
            except Exception as e:
                self.logger.error(f"Error loading Phone Detector: {e}")
                self.phone_detector = None
        
        # Summary
        loaded_count = len([d for d in [self.face_detector, self.face_matcher, self.eye_detector, self.phone_detector] if d is not None])
        self.logger.info(f"Detector initialization complete: {loaded_count} detector(s) loaded")
    
    def list_detectors(self):
        """
        List all registered detectors
        
        Returns:
            list: List of detector information dictionaries
        """
        detector_list = []
        for name, detector in self.detectors.items():
            detector_list.append({
                'name': name,
                'enabled': detector.enabled,
                'initialized': detector.initialized
            })
        return detector_list
    
    def get_detector(self, detector_name: str):
        """
        Get a detector by name
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            BaseDetector or None: The detector instance if found
        """
        return self.detectors.get(detector_name)
    
    def enable_detector(self, detector_name: str):
        """Enable a detector by name"""
        detector = self.get_detector(detector_name)
        if detector:
            detector.enable()
            return True
        return False
    
    def disable_detector(self, detector_name: str):
        """Disable a detector by name"""
        detector = self.get_detector(detector_name)
        if detector:
            detector.disable()
            return True
        return False

    
    def process_frame(self, frame):
        """
        Process frame with SEQUENTIAL pipeline:
        1. MediaPipe face detector gets faces
        2. Multiple faces alert
        3. If single face then verify
        4. After verify check eye movement
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with annotations
        """
        self.proctoring_results["total_frames_captured"] += 1
        self.frame_counter += 1
        
        # Check if we should process this frame
        if self.frame_counter <= self.frame_skip:
            # Skip processing, return frame as-is (no overlay if display is disabled)
            if getattr(self.config, 'DISPLAY_FEED', True):
                return self._add_skip_overlay(frame)
            else:
                return frame
        
        # Reset counter
        self.frame_counter = 0
        self.proctoring_results["total_frames_processed"] += 1
        self.session_logger.log_frame_processed()
        
        annotated_frame = frame.copy()
        
        # STEP 1: MediaPipe face detector gets faces
        face_meshes = []
        if self.face_detector and self.face_detector.enabled:
            try:
                face_meshes = self.face_detector.detect(frame)
            except Exception as e:
                self.logger.error(f"Error detecting faces: {e}")
                self.session_logger.log_alert('detection_error', f"Face detection failed: {e}", 'critical')
        
        num_faces = len(face_meshes)
        verification_result = None
        eye_result = None
        
        # STEP 2: Multiple faces alert
        if num_faces > 1:
            # Multiple faces detected - LOG ALERT & UPDATE ALERT STATE
            alert_msg = f"Multiple people detected: {num_faces} faces"
            
            self.session_logger.log_alert(
                'multiple_faces',
                alert_msg,
                'warning',
                {'num_faces': num_faces}
            )
            self.proctoring_results["alerts"].append({
                "timestamp": time.time(),
                "type": "multiple_faces",
                "message": alert_msg,
                "severity": "warning"
            })
            
            # Update alert communicator
            self.alert_comm.set_multiple_faces(True)
            self.alert_comm.set_no_face(False)  # Clear no face if multiple detected
        elif num_faces == 0:
            # No face detected - LOG ALERT & UPDATE ALERT STATE
            alert_msg = "No face detected"
            
            self.session_logger.log_alert(
                'no_face',
                alert_msg,
                'warning'
            )
            self.proctoring_results["alerts"].append({
                "timestamp": time.time(),
                "type": "no_face",
                "message": alert_msg,
                "severity": "warning"
            })
            
            # Update alert communicator
            self.alert_comm.set_no_face(True)
            self.alert_comm.set_multiple_faces(False)  # Clear multiple if none detected
        elif num_faces == 1:
            # Clear face count alerts (single face is good)
            self.alert_comm.set_no_face(False)
            self.alert_comm.set_multiple_faces(False)
            
            # STEP 3: If single face then verify
            if self.face_matcher and self.face_matcher.enabled:
                try:
                    verification_result = self._verify_face_sequential(frame, face_meshes[0])
                except Exception as e:
                    self.logger.error(f"Error during face verification: {e}")
            
            # STEP 4: After verify check eye movement
            eye_risk_detected = False
            if self.eye_detector and self.eye_detector.enabled:
                try:
                    # Use detect method to get eye data
                    eye_detections = self.eye_detector.detect(frame, face_meshes)
                    
                    # Process detections and calculate risk
                    if eye_detections:
                        for detection in eye_detections:
                            # Calculate risk for this eye
                            status, score, h_ratio, v_ratio = self.eye_detector.calculate_risk(detection)
                            
                            # Check for risk alerts
                            if "RISK" in status:
                                eye_risk_detected = True
                                alert_msg = f"Suspicious eye movement detected: {detection.get('eye_name', 'Unknown')} eye - {status}"
                                self.proctoring_results["alerts"].append({
                                    "timestamp": time.time(),
                                    "type": "eye_movement",
                                    "message": alert_msg,
                                    "severity": "warning"
                                })
                        
                        # Store detections for drawing
                        eye_result = eye_detections
                    
                    # Update alert communicator
                    self.alert_comm.set_eye_movement(eye_risk_detected)
                except Exception as e:
                    self.logger.error(f"Error during eye detection: {e}")
                    self.session_logger.log_alert('eye_detection_error', f"Eye detection failed: {e}", 'info')
            else:
                # Clear eye movement alert if detector disabled
                self.alert_comm.set_eye_movement(False)
        
        # STEP 5: Check for phone detection
        phone_detected = False
        if self.phone_detector and self.phone_detector.enabled:
            try:
                # Process frame for phone detection
                _, phone_result = self.phone_detector.process_frame(frame, draw=False)
                
                # Check if phone was detected (alert flag)
                if phone_result.get('alert', False):
                    phone_detected = True
                    
                    # Log critical alert
                    self.session_logger.log_alert(
                        'cheating_phone_detected',
                        'Phone detected',
                        'critical',
                        phone_result  # Metadata stored but not logged
                    )
                    self.proctoring_results["alerts"].append({
                        "timestamp": time.time(),
                        "type": "cheating_phone_detected",
                        "message": "Phone detected",
                        "severity": "critical"
                    })
                
                # Update alert communicator (critical alert, force=True)
                self.alert_comm.set_phone_detected(phone_detected)
            except Exception as e:
                self.logger.error(f"Error during phone detection: {e}")
                self.session_logger.log_alert('phone_detection_error', f"Phone detection failed: {e}", 'info')
        else:
            # Clear phone alert if detector disabled
            self.alert_comm.set_phone_detected(False)
        
        # Flush alert state to file if needed (debounced)
        self.alert_comm.flush_if_needed()
    
        # Only draw annotations if DISPLAY_FEED is enabled
        if getattr(self.config, 'DISPLAY_FEED', True):
            # Draw face meshes on frame with configuration
            if face_meshes and self.face_detector:
                show_all = getattr(self.config, 'SHOW_ALL_FACE_LANDMARKS', False)
                show_nums = getattr(self.config, 'SHOW_LANDMARK_NUMBERS', False)
                annotated_frame = self.face_detector.draw_faces(
                    annotated_frame, 
                    face_meshes,
                    show_all_landmarks=show_all,
                    show_landmark_numbers=show_nums
                )
            
            # Draw eye detection results if available
            if eye_result and self.eye_detector:
                try:
                    # Use process_frame to get annotated output
                    annotated_frame, _ = self.eye_detector.process_frame(annotated_frame, face_meshes, draw=True)
                except Exception as e:
                    self.logger.error(f"Error drawing eye keypoints: {e}")
            
            # Add unified status overlay (FPS, face status, phone status)
            annotated_frame = self._add_status_overlay(annotated_frame, num_faces, verification_result, phone_detected)
        
        return annotated_frame
    
    def _verify_face_sequential(self, frame, face_data):
        """
        Sequential face verification (no threading)
        
        Args:
            frame: Input frame
            face_data: Face mesh data
            
        Returns:
            dict: Verification result
        """
        try:
            bbox = face_data['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Match face
            result = self.face_matcher.match_with_details(face_roi)
            
            # Log result and update alert communicator if verification failed
            if not result.get('matched'):
                self.session_logger.log_alert(
                    'face_mismatch',
                    "Face verification failed",
                    'warning',
                    result  # Metadata stored but not logged
                )
                
                # Update alert communicator
                self.alert_comm.set_face_mismatch(True)
            else:
                # Clear face mismatch alert if face matched
                self.alert_comm.set_face_mismatch(False)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in face verification: {e}")
            return {'matched': False, 'error': str(e)}
    
    def _add_skip_overlay(self, frame):
        """Add overlay for skipped frames"""
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"Frame {self.proctoring_results['total_frames_captured']} (skipped)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        return overlay
    
    def _add_status_overlay(self, frame, num_faces, verification_result, phone_detected):
        """
        Add unified status overlay (FPS, Face Status, Phone Detection)
        Stacked vertically in top-left corner
        
        Args:
            frame: Input frame
            num_faces: Number of faces detected
            verification_result: Verification result from FaceMatcher
            phone_detected: Whether phone was detected
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        y_offset = 25
        line_height = 25
        font_scale = 0.5
        font_thickness = 1
        
        # Calculate FPS
        fps_text = "FPS: --"
        if hasattr(self, '_fps_start_time'):
            elapsed = time.time() - self._fps_start_time
            if elapsed > 0:
                fps = self._fps_frame_count / elapsed
                fps_text = f"FPS: {fps:.1f}"
        
        # 1. FPS Display
        cv2.putText(
            overlay,
            fps_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        y_offset += line_height
        
        # 2. Face Verification Status
        if num_faces > 1:
            face_text = f"ALERT: {num_faces} PEOPLE"
            face_color = (0, 0, 255)  # Red
        elif num_faces == 0:
            face_text = "NO FACE"
            face_color = (0, 0, 255)  # Red
        elif verification_result:
            if verification_result.get('matched'):
                face_text = "VERIFIED"
                face_color = (0, 255, 0)  # Green
                confidence = verification_result.get('confidence', 0)
                face_text += f" ({confidence:.0%})"
            else:
                face_text = "UNVERIFIED"
                face_color = (0, 165, 255)  # Orange
        else:
            face_text = "CHECKING..."
            face_color = (255, 255, 0)  # Yellow
        
        cv2.putText(
            overlay,
            face_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            face_color,
            font_thickness
        )
        y_offset += line_height
        
        # 3. Phone Detection Status
        if phone_detected:
            phone_text = "PHONE DETECTED!"
            phone_color = (0, 0, 255)  # Red
            cv2.putText(
                overlay,
                phone_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                phone_color,
                font_thickness
            )
        
        return overlay
    
    def _draw_all_detections(self, frame, frame_detections):
        """
        Draw all detection results on the frame
        
        Args:
            frame: Input frame
            frame_detections: Dictionary of detection results from all detectors
            
        Returns:
            Frame with all annotations drawn
        """
        annotated_frame = frame.copy()
        
        # Draw each detector's results
        for detector_name, results in frame_detections.items():
            detector = self.get_detector(detector_name)
            if not detector:
                continue
            
            try:
                # For face detector, draw faces with verification info
                if detector_name == "FaceDetector" and results.get("num_faces", 0) > 0:
                    face_objs = results.get("faces", [])
                    verification = results.get("verification")
                    
                    # Call detector's draw method
                    annotated_frame = detector.draw_faces(
                        annotated_frame,
                        face_objs,
                        verification_result=verification
                    )
                
                # For other detectors, check if they have detection results
                elif hasattr(detector, 'draw_detections') and results.get("detections"):
                    detections = results.get("detections", [])
                    annotated_frame = detector.draw_detections(annotated_frame, detections)
                    
            except Exception as e:
                self.logger.error(f"Error drawing detections for {detector_name}: {e}")
        
        return annotated_frame
    
    def _check_alerts(self, detector_name: str, detection_results: dict):
        """
        Check detection results for alert conditions
        Can be customized based on proctoring rules
        
        Args:
            detector_name: Name of the detector
            detection_results: Results from the detector
        """
        # Example: Alert if multiple faces detected
        if detector_name == "FaceDetector":
            num_faces = detection_results.get("num_faces", 0)
            
            if num_faces > 1:
                alert = {
                    "timestamp": time.time(),
                    "detector": detector_name,
                    "type": "multiple_faces",
                    "message": f"Multiple faces detected: {num_faces}",
                    "severity": "warning"
                }
                self.proctoring_results["alerts"].append(alert)
                self.logger.warning(f"ALERT: {alert['message']}")
            elif num_faces == 0:
                alert = {
                    "timestamp": time.time(),
                    "detector": detector_name,
                    "type": "no_face",
                    "message": "No face detected",
                    "severity": "warning"
                }
                self.proctoring_results["alerts"].append(alert)
                self.logger.warning(f"ALERT: {alert['message']}")
    
    def _add_proctoring_overlay(self, frame, frame_detections):
        """
        Add proctoring information overlay to frame
        
        Args:
            frame: Input frame
            frame_detections: Detection results for current frame
            
        Returns:
            Frame with overlay
        """
        overlay_frame = frame.copy()
        
        # Add detector status
        y_offset = 60
        for detector_name, results in frame_detections.items():
            status_text = f"{detector_name}: "
            
            if detector_name == "FaceDetector":
                num_faces = results.get("num_faces", 0)
                # Handle verification safely - it might be None
                verification = results.get("verification") or {}
                verified = verification.get("verified", False)
                
                if verified:
                    status_text += f"Verified ({num_faces} faces)"
                    color = (0, 255, 0)  # Green
                elif num_faces > 0:
                    status_text += f"Unverified ({num_faces} faces)"
                    color = (0, 165, 255)  # Orange
                else:
                    status_text += "No face detected"
                    color = (0, 0, 255)  # Red
            else:
                status_text += "Active"
                color = (255, 255, 255)  # White
            
            cv2.putText(
                overlay_frame,
                status_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            y_offset += 25
        
        # Add frame processing stats
        stats_text = f"Processed: {self.proctoring_results['total_frames_processed']}/{self.proctoring_results['total_frames_captured']}"
        cv2.putText(
            overlay_frame,
            stats_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return overlay_frame
    
    def get_proctoring_report(self):
        """
        Generate a comprehensive proctoring report
        
        Returns:
            dict: Proctoring statistics and results
        """
        report = {
            "session_stats": {
                "total_frames_captured": self.proctoring_results["total_frames_captured"],
                "total_frames_processed": self.proctoring_results["total_frames_processed"],
                "frame_skip": self.frame_skip,
                "processing_ratio": (
                    self.proctoring_results["total_frames_processed"] / 
                    self.proctoring_results["total_frames_captured"]
                    if self.proctoring_results["total_frames_captured"] > 0 else 0
                )
            },
            "detectors": self.list_detectors(),
            "alerts": self.proctoring_results["alerts"],
            "alert_summary": self._get_alert_summary()
        }
        
        return report
    
    def _get_alert_summary(self):
        """Generate summary of alerts"""
        summary = {}
        for alert in self.proctoring_results["alerts"]:
            alert_type = alert.get("type", "unknown")
            if alert_type not in summary:
                summary[alert_type] = 0
            summary[alert_type] += 1
        return summary
    
    def start(self):
        """
        Start the proctoring pipeline (initialize camera and display)
        
        Returns:
            bool: True if started successfully
        """
        return self.initialize()
    
    def stop(self):
        """
        Stop the proctoring pipeline (cleanup resources)
        """
        self.cleanup()
    
    def capture_frame(self):
        """
        Capture a frame from the camera
        
        Returns:
            Frame or None if capture failed
        """
        if not self.camera:
            return None
        
        success, frame = self.camera.read_frame()
        if not success:
            return None
        
        return frame
    
    def display_frame(self, frame):
        """
        Display a frame to the window
        
        Args:
            frame: Frame to display
        """
        # Skip display if DISPLAY_FEED is disabled or display not initialized
        if not self.display or not getattr(self.config, 'DISPLAY_FEED', True):
            return
        
        # Add FPS if enabled
        if self.config.SHOW_FPS:
            # Calculate FPS (simple moving average)
            if not hasattr(self, '_fps_start_time'):
                self._fps_start_time = time.time()
                self._fps_frame_count = 0
            
            self._fps_frame_count += 1
            elapsed = time.time() - self._fps_start_time
            
            if elapsed > 1.0:
                fps = self._fps_frame_count / elapsed
                self.display.show_frame(frame, fps=fps)
                self._fps_frame_count = 0
                self._fps_start_time = time.time()
            else:
                self.display.show_frame(frame)
        else:
            self.display.show_frame(frame)
    
    def cleanup(self):
        """Cleanup resources (override from CameraPipeline)"""
        print("\n[DEBUG] === ENTERING ProctorPipeline.cleanup() ===")
        print("[DEBUG] Step 0: Starting cleanup")
        
        try:
            print("[DEBUG] Step 1: Generating final report")
            # Generate final report
            self.logger.info("Generating final proctoring report...")
            report = self.get_proctoring_report()
            
            print("[DEBUG] Step 2: Logging report stats")
            self.logger.info("=" * 60)
            self.logger.info("PROCTORING SESSION REPORT")
            self.logger.info("=" * 60)
            self.logger.info(f"Total Frames Captured: {report['session_stats']['total_frames_captured']}")
            self.logger.info(f"Total Frames Processed: {report['session_stats']['total_frames_processed']}")
            self.logger.info(f"Processing Ratio: {report['session_stats']['processing_ratio']:.2%}")
            self.logger.info(f"Total Alerts: {len(report['alerts'])}")
            
            if report['alert_summary']:
                self.logger.info("\nAlert Summary:")
                for alert_type, count in report['alert_summary'].items():
                    self.logger.info(f"  - {alert_type}: {count}")
            
            self.logger.info("=" * 60)
            print("[DEBUG] Step 3: Report generation complete")
        except Exception as e:
            print(f"[DEBUG] ERROR in report generation: {e}")
            self.logger.error(f"Error generating final report: {e}")
        
        # Close session logger and save
        try:
            print("[DEBUG] Step 4: Checking session logger")
            if hasattr(self, 'session_logger') and self.session_logger:
                print("[DEBUG] Step 5: Getting session summary")
                summary = self.session_logger.get_session_summary()
                self.logger.info(f"Session logs saved to: {summary['log_file']}")
                self.logger.info(f"Session alerts saved to: {summary['alerts_file']}")
                print("[DEBUG] Step 6: Closing session logger")
                self.session_logger.close()
                print("[DEBUG] Step 7: Session logger closed")
        except Exception as e:
            print(f"[DEBUG] ERROR closing session logger: {e}")
            self.logger.error(f"Error closing session logger: {e}")
                # Force write final alert state and cleanup
        try:
            print("[DEBUG] Step 7.5: Cleaning up alert communicator")
            if hasattr(self, 'alert_comm') and self.alert_comm:
                print("[DEBUG] Forcing final alert state write")
                self.alert_comm.clear_all_alerts()  # Clear all alerts at end
                self.alert_comm.force_write()  # Force write final state
                print("[DEBUG] Alert communicator cleaned up")
                self.logger.info("Alert communicator state saved")
        except Exception as e:
            print(f"[DEBUG] ERROR cleaning up alert communicator: {e}")
            self.logger.error(f"Error cleaning up alert communicator: {e}")
                # Cleanup all registered detectors
        try:
            print("[DEBUG] Step 8: Checking detectors")
            if hasattr(self, 'detectors') and self.detectors:
                print(f"[DEBUG] Step 9: Found {len(self.detectors)} detectors to clean")
                self.logger.info("Cleaning up registered detectors...")
                # Create a list copy to avoid modifying dict during iteration
                detector_items = list(self.detectors.items())
                print(f"[DEBUG] Step 10: Created detector list copy with {len(detector_items)} items")
                
                for idx, (detector_name, detector) in enumerate(detector_items):
                    try:
                        print(f"[DEBUG] Step 11.{idx}: Cleaning detector '{detector_name}'")
                        if hasattr(detector, 'cleanup') and callable(detector.cleanup):
                            self.logger.info(f"Cleaning up {detector_name}...")
                            print(f"[DEBUG] Step 12.{idx}: Calling cleanup() on '{detector_name}'")
                            detector.cleanup()
                            print(f"[DEBUG] Step 13.{idx}: Cleanup completed for '{detector_name}'")
                        else:
                            print(f"[DEBUG] Step 12.{idx}: No cleanup method for '{detector_name}'")
                            self.logger.warning(f"{detector_name} has no cleanup method")
                    except Exception as e:
                        print(f"[DEBUG] ERROR cleaning up {detector_name}: {e}")
                        self.logger.error(f"Error cleaning up {detector_name}: {e}", exc_info=True)
                
                print("[DEBUG] Step 14: Clearing detectors dict")
                self.detectors.clear()
                print("[DEBUG] Step 15: Detectors cleared")
                self.logger.info("All detectors cleaned up")
        except Exception as e:
            print(f"[DEBUG] ERROR during detector cleanup: {e}")
            self.logger.error(f"Error during detector cleanup: {e}", exc_info=True)
        
        # Call parent cleanup (camera and display)
        try:
            print("[DEBUG] Step 16: Calling parent (CameraPipeline) cleanup")
            super().cleanup()
            print("[DEBUG] Step 17: Parent cleanup returned")
        except Exception as e:
            print(f"[DEBUG] ERROR in parent cleanup: {e}")
            self.logger.error(f"Error in parent cleanup: {e}", exc_info=True)
        
        print("[DEBUG] === EXITING ProctorPipeline.cleanup() ===")
