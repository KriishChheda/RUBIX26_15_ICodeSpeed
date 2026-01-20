"""
Proctor Logger Module
Handles session-based logging for proctoring events and alerts
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
import threading


class ProctorLogger:
    """
    Session-based logger for proctoring alerts and events
    Creates separate log files for each session identified by timestamp
    """
    
    def __init__(self, log_dir='logs/proctoring', session_id=None):
        """
        Initialize Proctor Logger
        
        Args:
            log_dir: Directory to store log files
            session_id: Optional session identifier (uses timestamp if None)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session ID from timestamp if not provided
        if session_id is None:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.session_id = session_id
        
        # Create log file paths
        self.log_file = self.log_dir / f"session_{self.session_id}.log"
        self.alerts_file = self.log_dir / f"session_{self.session_id}_alerts.json"
        
        # Session data
        self.session_start = datetime.now()
        self.session_data = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'end_time': None,
            'alerts': [],
            'statistics': {
                'total_frames': 0,
                'total_alerts': 0,
                'alert_types': {}
            }
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Configure file logger
        self.logger = logging.getLogger(f"ProctorLogger_{self.session_id}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Log session start
        self.logger.info("=" * 80)
        self.logger.info(f"PROCTORING SESSION STARTED - ID: {self.session_id}")
        self.logger.info("=" * 80)
        
        # Console logger for important alerts
        self.console_logger = logging.getLogger(__name__)
    
    def log_alert(self, alert_type, message, severity='warning', metadata=None):
        """
        Log an alert to the session
        
        Args:
            alert_type: Type of alert (e.g., 'multiple_faces', 'no_face', 'phone_detected')
            message: Alert message
            severity: Alert severity ('info', 'warning', 'critical')
            metadata: Additional metadata dictionary
        """
        with self.lock:
            timestamp = datetime.now()
            
            alert = {
                'timestamp': timestamp.isoformat(),
                'type': alert_type,
                'message': message,
                'severity': severity,
                'metadata': metadata or {}
            }
            
            # Add to session data
            self.session_data['alerts'].append(alert)
            self.session_data['statistics']['total_alerts'] += 1
            
            # Update alert type count
            if alert_type not in self.session_data['statistics']['alert_types']:
                self.session_data['statistics']['alert_types'][alert_type] = 0
            self.session_data['statistics']['alert_types'][alert_type] += 1
            
            # Log to file
            log_message = f"[{severity.upper()}] {alert_type}: {message}"
            if metadata:
                log_message += f" | {json.dumps(metadata)}"
            
            if severity == 'critical':
                self.logger.critical(log_message)
                self.console_logger.critical(log_message)
            elif severity == 'warning':
                self.logger.warning(log_message)
                self.console_logger.warning(log_message)
            else:
                self.logger.info(log_message)
    
    def log_info(self, message):
        """Log general information"""
        with self.lock:
            self.logger.info(message)
    
    def log_frame_processed(self):
        """Increment frame counter"""
        with self.lock:
            self.session_data['statistics']['total_frames'] += 1
    
    def log_detection(self, detector_name, results):
        """
        Log detection results
        
        Args:
            detector_name: Name of the detector
            results: Detection results dictionary
        """
        with self.lock:
            message = f"Detection [{detector_name}]: {json.dumps(results, default=str)}"
            self.logger.debug(message)
    
    def save_session(self):
        """Save session data to JSON file"""
        with self.lock:
            self.session_data['end_time'] = datetime.now().isoformat()
            
            try:
                with open(self.alerts_file, 'w') as f:
                    json.dump(self.session_data, f, indent=2)
                
                self.logger.info(f"Session data saved to: {self.alerts_file}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save session data: {e}")
                return False
    
    def get_session_summary(self):
        """
        Get session summary
        
        Returns:
            dict: Session statistics and summary
        """
        with self.lock:
            duration = (datetime.now() - self.session_start).total_seconds()
            
            return {
                'session_id': self.session_id,
                'duration_seconds': duration,
                'total_frames': self.session_data['statistics']['total_frames'],
                'total_alerts': self.session_data['statistics']['total_alerts'],
                'alert_types': self.session_data['statistics']['alert_types'],
                'log_file': str(self.log_file),
                'alerts_file': str(self.alerts_file)
            }
    
    def close(self):
        """Close logger and save session"""
        with self.lock:
            self.logger.info("=" * 80)
            self.logger.info("PROCTORING SESSION ENDED")
            self.logger.info(f"Duration: {(datetime.now() - self.session_start).total_seconds():.2f} seconds")
            self.logger.info(f"Total Frames: {self.session_data['statistics']['total_frames']}")
            self.logger.info(f"Total Alerts: {self.session_data['statistics']['total_alerts']}")
            
            if self.session_data['statistics']['alert_types']:
                self.logger.info("\nAlert Summary:")
                for alert_type, count in self.session_data['statistics']['alert_types'].items():
                    self.logger.info(f"  - {alert_type}: {count}")
            
            self.logger.info("=" * 80)
            
            # Save session data
            self.save_session()
            
            # Remove handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
