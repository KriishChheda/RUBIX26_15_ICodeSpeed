"""
Base Detector Module
Abstract base class for all detector modules in the proctoring system
"""

from abc import ABC, abstractmethod
import logging


class BaseDetector(ABC):
    """
    Abstract base class for all detector modules.
    All detector classes should inherit from this class.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize base detector
        
        Args:
            name: Name of the detector
            enabled: Whether the detector is enabled
        """
        self.name = name
        self.enabled = enabled
        self.initialized = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def load_model(self):
        """
        Load the model/detector. Must be implemented by subclasses.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame):
        """
        Process a frame and return detection results.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            tuple: (annotated_frame, detection_results)
                - annotated_frame: Frame with annotations drawn
                - detection_results: Dictionary containing detection information
        """
        pass
    
    def enable(self):
        """Enable the detector"""
        self.enabled = True
        self.logger.info(f"{self.name} detector enabled")
    
    def disable(self):
        """Disable the detector"""
        self.enabled = False
        self.logger.info(f"{self.name} detector disabled")
    
    def is_enabled(self):
        """Check if detector is enabled"""
        return self.enabled
    
    def is_initialized(self):
        """Check if detector is initialized"""
        return self.initialized
    
    def cleanup(self):
        """Cleanup detector resources. Override in subclasses if specific cleanup is needed."""
        self.logger.info(f"{self.name} detector cleanup complete (no resources to release)")
    
    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled}, initialized={self.initialized})"
