/**
 * Camera Monitor Spawner
 * Manages the background CV model for participant monitoring
 */

const path = require('path');
const fs = require('fs');
const { app } = require('electron');
const processManager = require('./processManager');
const logWatcher = require('./logWatcher');

class CameraMonitorSpawner {
  constructor() {
    this.monitorProcessId = null;
    this.sessionId = null;
    this.isMonitoring = false;
    this.pythonPath = null;
    this.scriptPath = null;
    this.participantImagePath = null;
  }

  /**
   * Initialize paths for Python and the proctoring script
   * @param {Object} config - Configuration object
   */
  initialize(config = {}) {
    // Determine Python path
    this.pythonPath = config.pythonPath || this.findPythonPath();
    
    // Determine script path (relative to app root)
    const appRoot = app.isPackaged 
      ? path.dirname(app.getPath('exe'))
      : path.join(__dirname, '../../../..');
    
    this.scriptPath = config.scriptPath || path.join(
      appRoot,
      'HD_ML_stuff',
      'proctor_main_background.py'
    );

    // Participant data path
    this.participantImagePath = config.participantImagePath || path.join(
      appRoot,
      'HD_ML_stuff',
      'data',
      'participant.png'
    );

    console.log('[CameraMonitor] Initialized with:');
    console.log(`  Python: ${this.pythonPath}`);
    console.log(`  Script: ${this.scriptPath}`);
    console.log(`  Participant Image: ${this.participantImagePath}`);

    return this.validatePaths();
  }

  /**
   * Find Python executable path
   * Looks for venv, conda, or system Python
   */
  findPythonPath() {
    const appRoot = app.isPackaged 
      ? path.dirname(app.getPath('exe'))
      : path.join(__dirname, '../../../..');

    // Check for virtual environment
    const venvPath = path.join(appRoot, 'HD_ML_stuff', '.venv', 'Scripts', 'python.exe');
    if (fs.existsSync(venvPath)) {
      return venvPath;
    }

    // Fallback to system Python
    return 'python';
  }

  /**
   * Validate that required paths exist
   */
  validatePaths() {
    const checks = {
      pythonExists: fs.existsSync(this.pythonPath) || this.pythonPath === 'python',
      scriptExists: fs.existsSync(this.scriptPath),
      participantImageExists: fs.existsSync(this.participantImagePath)
    };

    if (!checks.scriptExists) {
      console.error(`[CameraMonitor] Proctoring script not found: ${this.scriptPath}`);
    }

    return {
      success: checks.scriptExists,
      checks
    };
  }

  /**
   * Start the camera monitoring process
   * @param {Object} options - Monitor options
   * @returns {Object} Result with process info
   */
  startMonitoring(options = {}) {
    if (this.isMonitoring) {
      console.warn('[CameraMonitor] Monitoring already in progress');
      return {
        success: false,
        error: 'Monitoring already in progress',
        processId: this.monitorProcessId
      };
    }

    // Generate session ID
    this.sessionId = options.sessionId || `session_${Date.now()}`;

    // Prepare environment variables
    const env = {
      PROCTOR_SESSION_ID: this.sessionId,
      PROCTOR_PARTICIPANT_IMAGE: this.participantImagePath,
      PROCTOR_DISPLAY_FEED: 'false', // Background mode
      PROCTOR_FACE_DETECT: options.faceDetect !== false ? 'true' : 'false',
      PROCTOR_FACE_MATCH: options.faceMatch !== false ? 'true' : 'false',
      PROCTOR_EYE_TRACKING: options.eyeTracking !== false ? 'true' : 'false',
      PROCTOR_PHONE_DETECT: options.phoneDetect === true ? 'true' : 'false',
      ...options.env
    };

    // Prepare arguments
    const args = [
      this.scriptPath,
      ...(options.args || [])
    ];

    // Spawn the process
    const result = processManager.spawnProcess(this.pythonPath, args, {
      env,
      cwd: path.dirname(this.scriptPath),
      onStdout: (data) => {
        console.log(`[CameraMonitor] ${data.trim()}`);
        if (options.onOutput) {
          options.onOutput(data, 'stdout');
        }
      },
      onStderr: (data) => {
        console.error(`[CameraMonitor Error] ${data.trim()}`);
        if (options.onOutput) {
          options.onOutput(data, 'stderr');
        }
      },
      onExit: (code, signal) => {
        console.log(`[CameraMonitor] Process exited with code ${code}, signal ${signal}`);
        this.isMonitoring = false;
        this.monitorProcessId = null;
        if (options.onExit) {
          options.onExit(code, signal);
        }
      },
      onError: (error) => {
        console.error(`[CameraMonitor] Process error: ${error.message}`);
        this.isMonitoring = false;
        this.monitorProcessId = null;
        if (options.onError) {
          options.onError(error);
        }
      }
    });

    if (result.success) {
      this.monitorProcessId = result.processId;
      this.isMonitoring = true;
      
      console.log(`[CameraMonitor] Started monitoring - Process ID: ${result.processId}, PID: ${result.pid}`);
      
      // Start watching log files
      const logDir = this.getLogDirectory();
      if (logDir && options.watchLogs !== false) {
        // Give the process a moment to create the log file
        setTimeout(() => {
          logWatcher.watchSession(this.sessionId, path.dirname(logDir));
          
          // Forward log events to the callback if provided
          if (options.onLogAlert) {
            logWatcher.on('alert', ({ alert }) => {
              options.onLogAlert(alert);
            });
          }
          
          if (options.onLogNotification) {
            logWatcher.on('notification', ({ notification }) => {
              options.onLogNotification(notification);
            });
          }
        }, 1000);
      }
      
      return {
        success: true,
        processId: result.processId,
        pid: result.pid,
        sessionId: this.sessionId,
        logDirectory: logDir
      };
    } else {
      return {
        success: false,
        error: result.error
      };
    }
  }

  /**
   * Stop the camera monitoring process
   * @param {string} signal - Signal to send (default: 'SIGTERM')
   * @returns {Object} Result object
   */
  stopMonitoring(signal = 'SIGTERM') {
    if (!this.isMonitoring) {
      console.warn('[CameraMonitor] No monitoring in progress');
      return {
        success: false,
        error: 'No monitoring in progress'
      };
    }

    console.log(`[CameraMonitor] Stopping monitoring process ${this.monitorProcessId}`);
    
    // Stop watching logs
    const logDir = this.getLogDirectory();
    if (logDir && this.sessionId) {
      logWatcher.stopWatchingSession(this.sessionId, path.dirname(logDir));
    }
    
    const killed = processManager.killProcess(this.monitorProcessId, signal);
    
    if (killed) {
      // Give the process a moment to exit gracefully
      setTimeout(() => {
        // Force kill if still running
        if (processManager.isRunning(this.monitorProcessId)) {
          console.warn('[CameraMonitor] Process did not exit gracefully, force killing');
          processManager.killProcess(this.monitorProcessId, 'SIGKILL');
        }
      }, 3000);

      this.isMonitoring = false;
      const processId = this.monitorProcessId;
      this.monitorProcessId = null;

      return {
        success: true,
        processId
      };
    } else {
      return {
        success: false,
        error: 'Failed to kill process'
      };
    }
  }

  /**
   * Get current monitoring status
   * @returns {Object} Status object
   */
  getStatus() {
    if (!this.isMonitoring || !this.monitorProcessId) {
      return {
        isMonitoring: false,
        processId: null,
        sessionId: null
      };
    }

    const processInfo = processManager.getProcessInfo(this.monitorProcessId);
    
    return {
      isMonitoring: this.isMonitoring,
      processId: this.monitorProcessId,
      sessionId: this.sessionId,
      processInfo: processInfo
    };
  }

  /**
   * Get logs from the monitoring process
   * @param {string} type - 'stdout' or 'stderr'
   * @param {number} lines - Number of recent lines (0 = all)
   * @returns {string|null} Log output
   */
  getLogs(type = 'stdout', lines = 50) {
    if (!this.monitorProcessId) {
      return null;
    }

    if (type === 'stderr') {
      return processManager.getStderr(this.monitorProcessId, lines);
    } else {
      return processManager.getStdout(this.monitorProcessId, lines);
    }
  }

  /**
   * Set participant image for face matching
   * @param {string} imagePath - Path to participant image
   * @returns {boolean} Success status
   */
  setParticipantImage(imagePath) {
    if (!fs.existsSync(imagePath)) {
      console.error(`[CameraMonitor] Participant image not found: ${imagePath}`);
      return false;
    }

    this.participantImagePath = imagePath;
    console.log(`[CameraMonitor] Participant image set to: ${imagePath}`);
    return true;
  }

  /**
   * Get session log directory
   * @returns {string|null} Log directory path
   */
  getLogDirectory() {
    if (!this.sessionId) {
      return null;
    }

    const appRoot = app.isPackaged 
      ? path.dirname(app.getPath('exe'))
      : path.join(__dirname, '../../../..');

    return path.join(appRoot, 'HD_ML_stuff', 'logs', 'proctoring', this.sessionId);
  }
}

// Export singleton instance
module.exports = new CameraMonitorSpawner();
