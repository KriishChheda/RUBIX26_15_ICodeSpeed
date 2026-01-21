"""
Alert Communicator Module
Manages alert state communication between Python CV process and Electron frontend
"""

import json
import os
import time
import logging


class AlertCommunicator:
    """
    Manages alert state as a tuple of boolean values written to file.
    Provides debouncing and cooldown to prevent alert flapping.
    """
    
    # Alert type indices
    ALERT_PHONE = 0
    ALERT_NO_FACE = 1
    ALERT_MULTI_FACE = 2
    ALERT_FACE_MISMATCH = 3
    ALERT_EYE_MOVEMENT = 4
    
    def __init__(self, log_dir='logs/proctoring', alert_file_name='alert_state.txt', 
                 write_interval=0.1, cooldown_duration=1.0):
        """
        Initialize alert communicator
        
        Args:
            log_: Base directory for proctoring logs (default: 'logs/proctoring')
            alert_file_name: Name of alert state file (default: 'alert_state.txt')
            write_interval: Minimum seconds between file writes (default: 0.1)
            cooldown_duration: Seconds before same alert can retrigger (default: 1.0)
        """
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.alert_file = os.path.join(self.log_dir, alert_file_name)
        
        # Alert state: [phone, no_face, multi_face, face_mismatch, eye_movement]
        self.alert_state = [0, 0, 0, 0, 0]
        self.last_state = [0, 0, 0, 0, 0]
        
        # Write management
        self.last_write_time = 0
        self.write_interval = write_interval
        self.pending_write = False
        
        # Cooldown management (prevents rapid toggling)
        self.alert_cooldowns = [0.0] * 5  # Timestamp of last activation
        self.cooldown_duration = cooldown_duration
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize file with all zeros
        self._write_state()
        self.logger.info(f"Alert state file: {self.alert_file}")
    
    def set_alert(self, index, value, force=False):
        """
        Set alert state with cooldown protection
        
        Args:
            index: Alert index (0-4)
            value: Alert state (0 or 1)
            force: Bypass cooldown check (for critical alerts)
        
        Returns:
            bool: True if state was updated, False if blocked by cooldown
        """
        if index < 0 or index >= len(self.alert_state):
            self.logger.warning(f"Invalid alert index: {index}")
            return False
        
        # Normalize value to 0 or 1
        value = 1 if value else 0
        
        current_time = time.time()
        
        # Check cooldown (only for activation, not deactivation)
        if not force and value == 1:
            time_since_last = current_time - self.alert_cooldowns[index]
            if time_since_last < self.cooldown_duration:
                # Still in cooldown period
                return False
        
        # Update state if changed
        if self.alert_state[index] != value:
            old_value = self.alert_state[index]
            self.alert_state[index] = value
            self.pending_write = True
            
            # Update cooldown timestamp on activation
            if value == 1:
                self.alert_cooldowns[index] = current_time
                self.logger.debug(f"Alert {index} activated (was {old_value})")
            else:
                self.logger.debug(f"Alert {index} cleared (was {old_value})")
            
            return True
        
        return False
    
    def set_phone_detected(self, detected):
        """Set phone detection alert"""
        return self.set_alert(self.ALERT_PHONE, detected, force=True)  # Critical, bypass cooldown
    
    def set_no_face(self, detected):
        """Set no face detection alert"""
        return self.set_alert(self.ALERT_NO_FACE, detected)
    
    def set_multiple_faces(self, detected):
        """Set multiple faces alert"""
        return self.set_alert(self.ALERT_MULTI_FACE, detected)
    
    def set_face_mismatch(self, detected):
        """Set face verification failure alert"""
        return self.set_alert(self.ALERT_FACE_MISMATCH, detected)
    
    def set_eye_movement(self, detected):
        """Set suspicious eye movement alert"""
        return self.set_alert(self.ALERT_EYE_MOVEMENT, detected)
    
    def clear_alert(self, index):
        """Clear specific alert (bypass cooldown for clearing)"""
        if index < 0 or index >= len(self.alert_state):
            return False
        
        if self.alert_state[index] == 1:
            self.alert_state[index] = 0
            self.pending_write = True
            self.logger.debug(f"Alert {index} manually cleared")
            return True
        
        return False
    
    def clear_all_alerts(self):
        """Clear all alerts"""
        if any(self.alert_state):
            self.alert_state = [0, 0, 0, 0, 0]
            self.pending_write = True
            self.logger.debug("All alerts cleared")
            return True
        return False
    
    def flush_if_needed(self):
        """
        Write to file if state changed and enough time has passed.
        Call this in your main processing loop.
        
        Returns:
            bool: True if write occurred
        """
        if not self.pending_write:
            return False
        
        current_time = time.time()
        time_since_write = current_time - self.last_write_time
        
        if time_since_write >= self.write_interval:
            self._write_state()
            self.pending_write = False
            self.last_write_time = current_time
            return True
        
        return False
    
    def force_write(self):
        """
        Force immediate write regardless of interval.
        Use for session end or critical updates.
        """
        self._write_state()
        self.pending_write = False
        self.last_write_time = time.time()
    
    def _write_state(self):
        """
        Perform atomic write to alert state file.
        Uses temp file + rename for atomicity.
        """
        try:
            # Write to temporary file first
            temp_file = self.alert_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.alert_state, f)
            
            # Atomic rename (prevents partial reads)
            os.replace(temp_file, self.alert_file)
            
            # Log state changes
            if self.alert_state != self.last_state:
                self.logger.info(f"Alert state updated: {self.alert_state}")
                self.last_state = self.alert_state.copy()
        
        except Exception as e:
            self.logger.error(f"Error writing alert state: {e}")
    
    def get_current_state(self):
        """Get current alert state as list"""
        return self.alert_state.copy()
    
    def get_active_alerts(self):
        """
        Get list of currently active alert indices
        
        Returns:
            list: Indices of alerts that are active (value = 1)
        """
        return [i for i, val in enumerate(self.alert_state) if val == 1]
    
    def has_any_alerts(self):
        """Check if any alerts are active"""
        return any(self.alert_state)
    
    def get_alert_name(self, index):
        """Get human-readable name for alert index"""
        names = {
            self.ALERT_PHONE: "Phone Detected",
            self.ALERT_NO_FACE: "No Face",
            self.ALERT_MULTI_FACE: "Multiple Faces",
            self.ALERT_FACE_MISMATCH: "Face Mismatch",
            self.ALERT_EYE_MOVEMENT: "Eye Movement"
        }
        return names.get(index, f"Unknown Alert {index}")
    
    def __del__(self):
        """Ensure final state is written on cleanup"""
        try:
            if self.pending_write:
                self.force_write()
        except:
            pass
