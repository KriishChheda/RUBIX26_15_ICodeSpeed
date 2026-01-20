/**
 * Log File Watcher
 * Monitors proctoring log files for changes and emits new entries
 */

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const logParser = require('./logParser');

class LogFileWatcher extends EventEmitter {
  constructor() {
    super();
    this.watchers = new Map();
    this.filePositions = new Map();
    this.pollingIntervals = new Map();
  }

  /**
   * Start watching a log file
   * @param {string} filePath - Path to log file
   * @param {Object} options - Watch options
   * @returns {string} Watcher ID
   */
  watchFile(filePath, options = {}) {
    const {
      pollInterval = 500,  // Poll every 500ms
      parseLines = true,   // Parse log lines
      emitAlerts = true    // Emit parsed alerts
    } = options;

    if (this.watchers.has(filePath)) {
      console.warn(`[LogWatcher] Already watching ${filePath}`);
      return filePath;
    }

    // Initialize file position
    try {
      const stats = fs.statSync(filePath);
      this.filePositions.set(filePath, stats.size);
      console.log(`[LogWatcher] Starting to watch ${filePath} from position ${stats.size}`);
    } catch (error) {
      console.error(`[LogWatcher] Failed to get initial file size: ${error.message}`);
      this.filePositions.set(filePath, 0);
    }

    // Set up polling interval
    const interval = setInterval(() => {
      this.checkFileForUpdates(filePath, { parseLines, emitAlerts });
    }, pollInterval);

    this.pollingIntervals.set(filePath, interval);
    this.watchers.set(filePath, { filePath, options });

    console.log(`[LogWatcher] Now watching ${filePath}`);
    
    return filePath;
  }

  /**
   * Check file for new content
   * @param {string} filePath - Path to file
   * @param {Object} options - Check options
   */
  checkFileForUpdates(filePath, options = {}) {
    const { parseLines = true, emitAlerts = true } = options;

    try {
      const stats = fs.statSync(filePath);
      const currentSize = stats.size;
      const lastPosition = this.filePositions.get(filePath) || 0;

      // Check if file has new content
      if (currentSize > lastPosition) {
        const newBytes = currentSize - lastPosition;
        
        // Read new content
        const buffer = Buffer.alloc(newBytes);
        const fd = fs.openSync(filePath, 'r');
        fs.readSync(fd, buffer, 0, newBytes, lastPosition);
        fs.closeSync(fd);

        const newContent = buffer.toString('utf8');
        
        // Update position
        this.filePositions.set(filePath, currentSize);

        // Emit raw content
        this.emit('raw', { filePath, content: newContent, bytes: newBytes });

        // Parse and emit lines
        if (parseLines) {
          const lines = newContent.split('\n').filter(line => line.trim());
          
          for (const line of lines) {
            const parsed = logParser.parseLogLine(line);
            
            if (parsed) {
              this.emit('line', { filePath, line, parsed });

              // Emit alerts separately
              if (parsed.isAlert && emitAlerts) {
                this.emit('alert', { filePath, alert: parsed });
                
                // Convert to notification if applicable
                const notification = logParser.logToNotification(parsed);
                if (notification) {
                  this.emit('notification', { filePath, notification, parsed });
                }
              }
            }
          }
        }
      }
    } catch (error) {
      if (error.code === 'ENOENT') {
        // File doesn't exist yet, ignore
        return;
      }
      
      console.error(`[LogWatcher] Error checking file ${filePath}:`, error.message);
      this.emit('error', { filePath, error });
    }
  }

  /**
   * Stop watching a file
   * @param {string} filePath - Path to file
   */
  stopWatching(filePath) {
    const interval = this.pollingIntervals.get(filePath);
    if (interval) {
      clearInterval(interval);
      this.pollingIntervals.delete(filePath);
    }

    this.watchers.delete(filePath);
    this.filePositions.delete(filePath);

    console.log(`[LogWatcher] Stopped watching ${filePath}`);
  }

  /**
   * Stop watching all files
   */
  stopAll() {
    const filePaths = Array.from(this.watchers.keys());
    
    for (const filePath of filePaths) {
      this.stopWatching(filePath);
    }

    console.log(`[LogWatcher] Stopped watching all files`);
  }

  /**
   * Get list of watched files
   * @returns {Array<string>} File paths
   */
  getWatchedFiles() {
    return Array.from(this.watchers.keys());
  }

  /**
   * Check if file is being watched
   * @param {string} filePath - Path to file
   * @returns {boolean} True if watched
   */
  isWatching(filePath) {
    return this.watchers.has(filePath);
  }

  /**
   * Read entire log file and parse it
   * @param {string} filePath - Path to log file
   * @returns {Promise<Array>} Parsed log entries
   */
  async readAndParseFile(filePath) {
    try {
      const content = await fs.promises.readFile(filePath, 'utf8');
      return logParser.parseLogFile(content);
    } catch (error) {
      console.error(`[LogWatcher] Failed to read file ${filePath}:`, error.message);
      return [];
    }
  }

  /**
   * Read and parse alerts JSON file
   * @param {string} filePath - Path to JSON file
   * @returns {Promise<Object|null>} Parsed alerts data
   */
  async readAlertsJSON(filePath) {
    try {
      const content = await fs.promises.readFile(filePath, 'utf8');
      return logParser.parseAlertsJSON(content);
    } catch (error) {
      console.error(`[LogWatcher] Failed to read alerts JSON ${filePath}:`, error.message);
      return null;
    }
  }

  /**
   * Watch both log file and alerts JSON
   * @param {string} sessionId - Session ID
   * @param {string} logDir - Log directory
   * @returns {Object} Watcher IDs
   */
  watchSession(sessionId, logDir) {
    const logFile = path.join(logDir, `session_${sessionId}.log`);
    const alertsFile = path.join(logDir, `session_${sessionId}_alerts.json`);

    const logWatcherId = this.watchFile(logFile, {
      pollInterval: 500,
      parseLines: true,
      emitAlerts: true
    });

    console.log(`[LogWatcher] Watching session ${sessionId}`);
    console.log(`  Log: ${logFile}`);
    console.log(`  Alerts: ${alertsFile}`);

    return {
      sessionId,
      logFile,
      alertsFile,
      logWatcherId
    };
  }

  /**
   * Stop watching a session
   * @param {string} sessionId - Session ID
   * @param {string} logDir - Log directory
   */
  stopWatchingSession(sessionId, logDir) {
    const logFile = path.join(logDir, `session_${sessionId}.log`);
    this.stopWatching(logFile);
    
    console.log(`[LogWatcher] Stopped watching session ${sessionId}`);
  }
}

// Export singleton instance
module.exports = new LogFileWatcher();
