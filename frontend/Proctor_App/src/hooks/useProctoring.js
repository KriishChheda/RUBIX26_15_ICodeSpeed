import { useState, useCallback, useEffect } from 'react';
import { useNotification } from './useNotification';

/**
 * Custom hook for managing proctoring system
 * Provides interface to start/stop camera monitoring from React components
 */
export const useProctoring = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [monitoringStatus, setMonitoringStatus] = useState(null);
  const [error, setError] = useState(null);
  const [processId, setProcessId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const { notify } = useNotification();

  /**
   * Check if window.electron is available
   */
  const isElectron = typeof window !== 'undefined' && window.electron;

  /**
   * Start proctoring monitoring
   * @param {Object} options - Monitoring options
   * @returns {Promise<Object>} Result object
   */
  const startMonitoring = useCallback(async (options = {}) => {
    if (!isElectron) {
      const err = 'Proctoring is only available in Electron app';
      setError(err);
      return { success: false, error: err };
    }

    try {
      setError(null);
      
      const defaultOptions = {
        sessionId: `session_${Date.now()}`,
        faceDetect: true,
        faceMatch: true,
        eyeTracking: true,
        phoneDetect: false,
        config: {}
      };

      const finalOptions = { ...defaultOptions, ...options };

      const result = await window.electron.invoke('proctoring:start', finalOptions);

      if (result.success) {
        setIsMonitoring(true);
        setProcessId(result.processId);
        setSessionId(result.sessionId);
        setMonitoringStatus('running');
      } else {
        setError(result.error || 'Failed to start monitoring');
      }

      return result;
    } catch (err) {
      const errorMsg = err.message || 'Failed to start monitoring';
      setError(errorMsg);
      setIsMonitoring(false);
      return { success: false, error: errorMsg };
    }
  }, [isElectron]);

  /**
   * Stop proctoring monitoring
   * @returns {Promise<Object>} Result object
   */
  const stopMonitoring = useCallback(async () => {
    if (!isElectron) {
      const err = 'Proctoring is only available in Electron app';
      setError(err);
      return { success: false, error: err };
    }

    try {
      setError(null);
      
      const result = await window.electron.invoke('proctoring:stop');

      if (result.success) {
        setIsMonitoring(false);
        setProcessId(null);
        setMonitoringStatus('stopped');
      } else {
        setError(result.error || 'Failed to stop monitoring');
      }

      return result;
    } catch (err) {
      const errorMsg = err.message || 'Failed to stop monitoring';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    }
  }, [isElectron]);

  /**
   * Get current monitoring status
   * @returns {Promise<Object>} Status object
   */
  const getStatus = useCallback(async () => {
    if (!isElectron) {
      return { isMonitoring: false, error: 'Not in Electron' };
    }

    try {
      const status = await window.electron.invoke('proctoring:status');
      
      setIsMonitoring(status.isMonitoring);
      setProcessId(status.processId);
      setSessionId(status.sessionId);
      setMonitoringStatus(status.isMonitoring ? 'running' : 'stopped');
      
      return status;
    } catch (err) {
      console.error('Failed to get proctoring status:', err);
      return { isMonitoring: false, error: err.message };
    }
  }, [isElectron]);

  /**
   * Get logs from monitoring process
   * @param {string} type - 'stdout' or 'stderr'
   * @param {number} lines - Number of recent lines
   * @returns {Promise<string|null>} Log output
   */
  const getLogs = useCallback(async (type = 'stdout', lines = 50) => {
    if (!isElectron) {
      return null;
    }

    try {
      const result = await window.electron.invoke('proctoring:getLogs', type, lines);
      return result.success ? result.logs : null;
    } catch (err) {
      console.error('Failed to get logs:', err);
      return null;
    }
  }, [isElectron]);

  /**
   * Set participant image for face matching
   * @param {string} imagePath - Path to participant image
   * @returns {Promise<Object>} Result object
   */
  const setParticipantImage = useCallback(async (imagePath) => {
    if (!isElectron) {
      return { success: false, error: 'Not in Electron' };
    }

    try {
      const result = await window.electron.invoke('proctoring:setParticipantImage', imagePath);
      return result;
    } catch (err) {
      console.error('Failed to set participant image:', err);
      return { success: false, error: err.message };
    }
  }, [isElectron]);

  /**
   * Get alerts JSON data for a session
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object|null>} Alerts data
   */
  const getAlertsJSON = useCallback(async (sessionId) => {
    if (!isElectron) {
      return null;
    }

    try {
      const result = await window.electron.invoke('proctoring:getAlertsJSON', sessionId);
      return result.success ? result.data : null;
    } catch (err) {
      console.error('Failed to get alerts JSON:', err);
      return null;
    }
  }, [isElectron]);

  /**
   * Listen for proctoring events from main process
   */
  useEffect(() => {
    if (!isElectron || !window.electron.receive) {
      return;
    }

    // Listen for monitoring stopped event
    window.electron.receive('proctoring:stopped', ({ code, signal }) => {
      console.log(`Proctoring stopped: code=${code}, signal=${signal}`);
      setIsMonitoring(false);
      setProcessId(null);
      setMonitoringStatus('stopped');
    });

    // Listen for output events if needed
    window.electron.receive('proctoring:output', ({ data, type }) => {
      console.log(`[Proctoring ${type}]:`, data);
    });

    // Listen for parsed alerts
    window.electron.receive('proctoring:alert', (alert) => {
      console.log('[Proctoring Alert] RECEIVED:', JSON.stringify(alert, null, 2));
      console.log('[Proctoring Alert] notify object:', notify);
      
      // Display notification immediately when alert is received
      // The alert already contains the fixed message and category
      if (notify && alert) {
        const { severity, message, category } = alert;
        
        console.log(`[Proctoring Alert] Calling notify.${severity === 'critical' ? 'error' : 'warning'}("${message}", category: "${category}")`);
        
        // Map severity to notification type
        const notificationType = severity === 'critical' ? 'error' : 'warning';
        
        // Display notification
        if (notificationType === 'error') {
          notify.error(message, 5000, category); // Critical alerts stay longer (5s)
        } else {
          notify.warning(message, 3000, category); // Warning alerts (3s)
        }
        
        console.log('[Proctoring Alert] Notification displayed');
      } else {
        console.warn('[Proctoring Alert] Cannot display - notify:', !!notify, 'alert:', !!alert);
      }
    });

    // Listen for notifications and display them
    window.electron.receive('proctoring:notification', (notification) => {
      console.log('[Proctoring Notification] RECEIVED:', JSON.stringify(notification, null, 2));
      console.log('[Proctoring Notification] notify object:', notify);
      
      // Display notification using the notification system
      if (notify && notification.notification) {
        const { type, message, category } = notification.notification;
        
        console.log(`[Proctoring Notification] Calling notify.${type}("${message}", category: "${category}")`);
        
        // Use category for deduplication (e.g., 'no_face', 'eye_movement', etc.)
        // This prevents spam of the same alert type
        
        // Use the appropriate notification method based on type
        if (type === 'error') {
          notify.error(message, 3000, category);
        } else if (type === 'warning') {
          notify.warning(message, 3000, category);
        } else if (type === 'success') {
          notify.success(message, 3000, category);
        } else {
          notify.info(message, 3000, category);
        }
        
        console.log('[Proctoring Notification] Notification displayed');
      } else {
        console.warn('[Proctoring Notification] Cannot display - notify:', !!notify, 'notification.notification:', !!(notification && notification.notification));
      }
    });
  }, [isElectron, notify]);

  /**
   * Poll status on mount to sync state
   */
  useEffect(() => {
    if (isElectron) {
      getStatus();
    }
  }, [isElectron, getStatus]);

  return {
    // State
    isMonitoring,
    monitoringStatus,
    error,
    processId,
    sessionId,
    isElectron,
    
    // Methods
    startMonitoring,
    stopMonitoring,
    getStatus,
    getLogs,
    setParticipantImage,
    getAlertsJSON
  };
};
