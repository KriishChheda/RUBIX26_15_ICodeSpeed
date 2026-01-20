"""
Proctor Module
Main proctoring pipeline that orchestrates multiple detection modules
"""

import logging
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .camera_pipeline import CameraPipeline
from .base_detector import BaseDetector


class ProctorPipeline(CameraPipeline):
    """
    Proctoring pipeline that inherits from CameraPipeline.
    Manages multiple detector modules and processes frames at a configurable rate.
    """
    
    def __init__(self, config=None, frame_skip=2):
        """
        Initialize proctoring pipeline
        
        Args:
            config: Configuration object (uses default Config if None)
            frame_skip: Number of frames to skip between processing (default: 2, process every 3rd frame)
        """
        super().__init__(config)
        
        # Detector management
        self.detectors = []
        self.frame_skip = frame_skip
        self.frame_counter = 0
        
        # Thread pool for parallel detector execution
        self.max_workers = 4  # Maximum number of parallel detector threads
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Proctoring state
        self.proctoring_results = {
            "total_frames_captured": 0,
            "total_frames_processed": 0,
            "detections": {},
            "alerts": []
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Proctoring pipeline initialized with frame_skip={frame_skip}, parallel processing enabled")
    
    def register_detector(self, detector: BaseDetector):
        """
        Register a detector module to the proctoring pipeline
        
        Args:
            detector: An instance of BaseDetector or its subclass
            
        Returns:
            bool: True if detector was registered successfully
        """
        if not isinstance(detector, BaseDetector):
            self.logger.error(f"Detector must be an instance of BaseDetector, got {type(detector)}")
            return False
        
        # Initialize detector if not already initialized
        if not detector.is_initialized():
            self.logger.info(f"Initializing detector: {detector.name}")
            if not detector.load_model():
                self.logger.error(f"Failed to initialize detector: {detector.name}")
                return False
        
        self.detectors.append(detector)
        self.proctoring_results["detections"][detector.name] = []
        self.logger.info(f"Registered detector: {detector.name}")
        return True
    
    def unregister_detector(self, detector_name: str):
        """
        Unregister a detector module by name
        
        Args:
            detector_name: Name of the detector to remove
            
        Returns:
            bool: True if detector was removed
        """
        for i, detector in enumerate(self.detectors):
            if detector.name == detector_name:
                self.detectors.pop(i)
                self.logger.info(f"Unregistered detector: {detector_name}")
                return True
        
        self.logger.warning(f"Detector not found: {detector_name}")
        return False
    
    def get_detector(self, detector_name: str):
        """
        Get a detector by name
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            BaseDetector or None: The detector instance if found
        """
        for detector in self.detectors:
            if detector.name == detector_name:
                return detector
        return None
    
    def list_detectors(self):
        """
        Get list of all registered detectors
        
        Returns:
            list: List of detector names and their status
        """
        return [
            {
                "name": d.name,
                "enabled": d.is_enabled(),
                "initialized": d.is_initialized()
            }
            for d in self.detectors
        ]
    
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
    
    def _run_detector(self, detector, frame):
        """
        Run a single detector on a frame (executed in parallel thread)
        
        Args:
            detector: Detector instance to run
            frame: Input frame
            
        Returns:
            tuple: (detector_name, detection_results, processing_time)
        """
        try:
            start_time = time.time()
            # Each detector gets its own copy of the frame for thread safety
            frame_copy = frame.copy()
            
            # Check if this is FaceDetector and verification is enabled
            if detector.name == "FaceDetector" and hasattr(self.config, 'FACE_VERIFICATION_ENABLED'):
                verify_enabled = self.config.FACE_VERIFICATION_ENABLED
                _, detection_results = detector.process_frame(frame_copy, verify=verify_enabled, draw=False)
            else:
                _, detection_results = detector.process_frame(frame_copy, draw=False)
            
            processing_time = time.time() - start_time
            
            detection_results["processing_time_ms"] = processing_time * 1000
            detection_results["timestamp"] = time.time()
            
            return detector.name, detection_results, None
            
        except Exception as e:
            self.logger.error(f"Error processing frame with {detector.name}: {e}", exc_info=True)
            return detector.name, None, str(e)
    
    def process_frame(self, frame):
        """
        Process frame with all registered detectors in parallel (overrides CameraPipeline.process_frame)
        Only processes every Nth frame based on frame_skip setting
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with annotations
        """
        self.proctoring_results["total_frames_captured"] += 1
        self.frame_counter += 1
        
        # Check if we should process this frame
        if self.frame_counter <= self.frame_skip:
            # Skip processing, just return the frame
            return frame
        
        # Reset counter
        self.frame_counter = 0
        self.proctoring_results["total_frames_processed"] += 1
        
        # Get enabled detectors
        enabled_detectors = [d for d in self.detectors if d.is_enabled()]
        
        if not enabled_detectors:
            return frame
        
        # Run all detectors in parallel using thread pool
        frame_detections = {}
        futures = {}
        
        for detector in enabled_detectors:
            future = self.executor.submit(self._run_detector, detector, frame)
            futures[future] = detector.name
        
        # Collect results as they complete
        for future in as_completed(futures):
            detector_name, detection_results, error = future.result()
            
            if error:
                self.logger.error(f"Detector {detector_name} failed: {error}")
                continue
            
            if detection_results:
                frame_detections[detector_name] = detection_results
                
                # Add to history
                self.proctoring_results["detections"][detector_name].append(detection_results)
                
                # Check for alerts
                self._check_alerts(detector_name, detection_results)
        
        # Now draw all detection results on the frame sequentially
        processed_frame = self._draw_all_detections(frame, frame_detections)
        
        # Add proctoring info overlay
        processed_frame = self._add_proctoring_overlay(processed_frame, frame_detections)
        
        return processed_frame
    
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
            
            # Check for spoofing attempt
            verification = detection_results.get("verification")
            if verification and verification.get("alert") == "spoofing_attempt":
                alert = {
                    "timestamp": time.time(),
                    "detector": detector_name,
                    "type": "spoofing_attempt",
                    "message": f"SPOOFING DETECTED - Photo/video attack! Score: {verification.get('spoof_score')}",
                    "severity": "critical"
                }
                self.proctoring_results["alerts"].append(alert)
                self.logger.error(f"CRITICAL ALERT: {alert['message']}")
            
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
                is_real = verification.get("is_real", True)
                
                # Check for spoofing
                if not is_real:
                    status_text += "SPOOF DETECTED!"
                    color = (0, 0, 255)  # Red - critical
                elif verified:
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
    
    def cleanup(self):
        """Cleanup resources (override from CameraPipeline)"""
        # Shutdown thread pool
        self.logger.info("Shutting down detector thread pool...")
        self.executor.shutdown(wait=True)
        
        # Generate final report
        self.logger.info("Generating final proctoring report...")
        report = self.get_proctoring_report()
        
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
        
        # Call parent cleanup
        super().cleanup()
