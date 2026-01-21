/**
 * Camera Monitor Spawner
 * Manages the background CV model for participant monitoring
 */

const path = require('path');
const fs = require('fs');
const { app } = require('electron');
const processManager = require('./processManager');
const logWatcher = require('./logWatcher');
const alertStateWatcher = require('./alertStateWatcher');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '../../.env') });

class CameraMonitorSpawner {
  constructor() {
    this.monitorProcessId = null;
    this.sessionId = null;
    this.isMonitoring = false;
    this.pythonPath = null;
    this.scriptPath = null;
    this.participantImagePath = null;
    
    // Environment config
    this.projectPath = process.env.PYTHON_PROJECT_PATH || 'Mustan_ML_stuff';
    this.scriptName = process.env.PYTHON_SCRIPT_NAME || 'proctor_main_background.py';
    this.participantImageRelPath = process.env.PARTICIPANT_IMAGE_PATH || 'data/participant.png';
    this.logDirPath = process.env.LOG_DIRECTORY_PATH || 'logs/proctoring';
    this.alertStateFilename = process.env.ALERT_STATE_FILENAME || 'alert_state.txt';
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
      this.projectPath,
      this.scriptName
    );

    // Participant data path
    this.participantImagePath = config.participantImagePath || path.join(
      appRoot,
      this.projectPath,
      this.participantImageRelPath
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
   * Supports Windows, macOS, and Linux
   */
  findPythonPath() {
    const appRoot = app.isPackaged 
      ? path.dirname(app.getPath('exe'))
      : path.join(__dirname, '../../../..');

    const platform = process.platform;
    const pythonExeName = platform === 'win32' ? 'python.exe' : 'python';
    const venvBinDir = platform === 'win32' ? 'Scripts' : 'bin';

    // Priority order for finding Python:
    // 1. Virtual environment in project/.venv
    // 2. Virtual environment in project/venv
    // 3. Conda environment (if CONDA_PREFIX is set)
    // 4. System Python

    // Check for .venv (common convention)
    const venvPath1 = path.join(appRoot, this.projectPath, '.venv', venvBinDir, pythonExeName);
    if (fs.existsSync(venvPath1)) {
      console.log(`[CameraMonitor] Found venv Python: ${venvPath1}`);
      return venvPath1;
    }

    // Check for venv (without dot)
    const venvPath2 = path.join(appRoot, this.projectPath, 'venv', venvBinDir, pythonExeName);
    if (fs.existsSync(venvPath2)) {
      console.log(`[CameraMonitor] Found venv Python: ${venvPath2}`);
      return venvPath2;
    }

    // Check for conda environment
    if (process.env.CONDA_PREFIX) {
      const condaPath = path.join(process.env.CONDA_PREFIX, venvBinDir, pythonExeName);
      if (fs.existsSync(condaPath)) {
        console.log(`[CameraMonitor] Found conda Python: ${condaPath}`);
        return condaPath;
      }
    }

    // Check for env folder (another common name)
    const envPath = path.join(appRoot, this.projectPath, 'env', venvBinDir, pythonExeName);
    if (fs.existsSync(envPath)) {
      console.log(`[CameraMonitor] Found env Python: ${envPath}`);
      return envPath;
    }

    // Fallback to system Python
    console.log('[CameraMonitor] Using system Python');
    return pythonExeName;
  }

  /**
   * Validate that required paths exist
   */
  validatePaths() {
    const pythonExeName = process.platform === 'win32' ? 'python.exe' : 'python';
    
    const checks = {
      pythonExists: fs.existsSync(this.pythonPath) || this.pythonPath === pythonExeName,
      scriptExists: fs.existsSync(this.scriptPath),
      participantImageExists: fs.existsSync(this.participantImagePath)
    };

    if (!checks.pythonExists && this.pythonPath !== pythonExeName) {
      console.error(`[CameraMonitor] Python executable not found: ${this.pythonPath}`);
    }

    if (!checks.scriptExists) {
      console.error(`[CameraMonitor] Proctoring script not found: ${this.scriptPath}`);
    }

    if (!checks.participantImageExists) {
      console.warn(`[CameraMonitor] Participant image not found: ${this.participantImagePath}`);
      console.warn(`[CameraMonitor] Face matching will be disabled`);
    }

    return {
      success: checks.scriptExists && (checks.pythonExists || this.pythonPath === pythonExeName),
      checks,
      warnings: !checks.participantImageExists ? ['Participant image not found'] : []
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
      
      // Start watching alert state file
      const logDir = this.getLogDirectory();
      if (logDir && options.watchAlerts !== false) {
        // Give the process a moment to create the alert state file
        setTimeout(() => {
          const alertStateFile = path.join(logDir, this.alertStateFilename);
          const pollInterval = parseInt(process.env.ALERT_POLL_INTERVAL || '200');
          alertStateWatcher.startWatching(alertStateFile, { pollInterval });
          
          // Forward alert events to the callback if provided
          if (options.onLogAlert) {
            alertStateWatcher.on('alert', (alert) => {
              options.onLogAlert(alert);
            });
          }
          
          if (options.onLogNotification) {
            alertStateWatcher.on('notification', (notification) => {
              options.onLogNotification(notification);
            });
          }

          console.log(`[CameraMonitor] Watching alert state file: ${alertStateFile}`);
        }, 1000);
      }

      // Optionally still watch log files for historical data
      if (logDir && options.watchLogs === true) {
        setTimeout(() => {
          logWatcher.watchSession(this.sessionId, path.dirname(logDir));
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
    
    // Stop watching alert state file
    alertStateWatcher.stopWatching();
    
    // Stop watching logs if enabled
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
    // Check if projectPath is absolute or relative
    const isAbsolute = path.isAbsolute(this.projectPath);
    
    if (isAbsolute) {
      // If projectPath is absolute, use it directly
      return path.join(this.projectPath, this.logDirPath);
    } else {
      // If projectPath is relative, join with appRoot
      const appRoot = app.isPackaged 
        ? path.dirname(app.getPath('exe'))
        : path.join(__dirname, '../../../..');
      
      return path.join(appRoot, this.projectPath, this.logDirPath);
    }
  }
}

// Export singleton instance
module.exports = new CameraMonitorSpawner();
