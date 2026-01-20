/**
 * Process Management Utilities for Electron
 * Handles spawning and managing background processes
 */

const { spawn } = require('child_process');
const path = require('path');
const { app } = require('electron');

class ProcessManager {
  constructor() {
    this.processes = new Map();
    this.processIdCounter = 0;
  }

  /**
   * Spawn a new background process
   * @param {string} command - Command to execute
   * @param {Array<string>} args - Arguments for the command
   * @param {Object} options - Spawn options
   * @returns {Object} Process info with id and handle
   */
  spawnProcess(command, args = [], options = {}) {
    const processId = ++this.processIdCounter;
    
    const defaultOptions = {
      cwd: options.cwd || app.getAppPath(),
      env: { ...process.env, ...options.env },
      detached: false,
      stdio: options.stdio || ['ignore', 'pipe', 'pipe']
    };

    const finalOptions = { ...defaultOptions, ...options };

    try {
      const childProcess = spawn(command, args, finalOptions);
      
      const processInfo = {
        id: processId,
        process: childProcess,
        pid: childProcess.pid,
        command,
        args,
        startTime: Date.now(),
        status: 'running',
        stdout: [],
        stderr: [],
        exitCode: null
      };

      // Capture stdout
      if (childProcess.stdout) {
        childProcess.stdout.on('data', (data) => {
          const output = data.toString();
          processInfo.stdout.push(output);
          if (options.onStdout) {
            options.onStdout(output);
          }
        });
      }

      // Capture stderr
      if (childProcess.stderr) {
        childProcess.stderr.on('data', (data) => {
          const error = data.toString();
          processInfo.stderr.push(error);
          if (options.onStderr) {
            options.onStderr(error);
          }
        });
      }

      // Handle process exit
      childProcess.on('exit', (code, signal) => {
        processInfo.status = 'exited';
        processInfo.exitCode = code;
        processInfo.exitSignal = signal;
        processInfo.endTime = Date.now();
        
        if (options.onExit) {
          options.onExit(code, signal);
        }
      });

      // Handle errors
      childProcess.on('error', (error) => {
        processInfo.status = 'error';
        processInfo.error = error.message;
        
        if (options.onError) {
          options.onError(error);
        }
      });

      this.processes.set(processId, processInfo);
      
      console.log(`[ProcessManager] Spawned process ${processId} (PID: ${childProcess.pid}): ${command} ${args.join(' ')}`);
      
      return {
        success: true,
        processId,
        pid: childProcess.pid
      };
      
    } catch (error) {
      console.error(`[ProcessManager] Failed to spawn process: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get information about a running process
   * @param {number} processId - Process ID
   * @returns {Object|null} Process information
   */
  getProcessInfo(processId) {
    const processInfo = this.processes.get(processId);
    if (!processInfo) {
      return null;
    }

    return {
      id: processInfo.id,
      pid: processInfo.pid,
      command: processInfo.command,
      args: processInfo.args,
      status: processInfo.status,
      startTime: processInfo.startTime,
      endTime: processInfo.endTime,
      exitCode: processInfo.exitCode,
      error: processInfo.error,
      uptime: processInfo.endTime 
        ? processInfo.endTime - processInfo.startTime 
        : Date.now() - processInfo.startTime
    };
  }

  /**
   * Get stdout output from a process
   * @param {number} processId - Process ID
   * @param {number} lines - Number of recent lines to return (0 = all)
   * @returns {string|null} Output or null if process not found
   */
  getStdout(processId, lines = 0) {
    const processInfo = this.processes.get(processId);
    if (!processInfo) {
      return null;
    }

    if (lines === 0) {
      return processInfo.stdout.join('');
    }

    return processInfo.stdout.slice(-lines).join('');
  }

  /**
   * Get stderr output from a process
   * @param {number} processId - Process ID
   * @param {number} lines - Number of recent lines to return (0 = all)
   * @returns {string|null} Error output or null if process not found
   */
  getStderr(processId, lines = 0) {
    const processInfo = this.processes.get(processId);
    if (!processInfo) {
      return null;
    }

    if (lines === 0) {
      return processInfo.stderr.join('');
    }

    return processInfo.stderr.slice(-lines).join('');
  }

  /**
   * Check if a process is running
   * @param {number} processId - Process ID
   * @returns {boolean} True if running
   */
  isRunning(processId) {
    const processInfo = this.processes.get(processId);
    return processInfo && processInfo.status === 'running';
  }

  /**
   * Kill a running process
   * @param {number} processId - Process ID
   * @param {string} signal - Signal to send (default: 'SIGTERM')
   * @returns {boolean} True if process was killed
   */
  killProcess(processId, signal = 'SIGTERM') {
    const processInfo = this.processes.get(processId);
    
    if (!processInfo) {
      console.warn(`[ProcessManager] Process ${processId} not found`);
      return false;
    }

    if (processInfo.status !== 'running') {
      console.warn(`[ProcessManager] Process ${processId} is not running (status: ${processInfo.status})`);
      return false;
    }

    try {
      processInfo.process.kill(signal);
      console.log(`[ProcessManager] Sent ${signal} to process ${processId} (PID: ${processInfo.pid})`);
      return true;
    } catch (error) {
      console.error(`[ProcessManager] Failed to kill process ${processId}: ${error.message}`);
      return false;
    }
  }

  /**
   * Kill all running processes
   * @param {string} signal - Signal to send (default: 'SIGTERM')
   */
  killAll(signal = 'SIGTERM') {
    const runningProcesses = Array.from(this.processes.entries())
      .filter(([_, info]) => info.status === 'running');

    console.log(`[ProcessManager] Killing ${runningProcesses.length} running process(es)`);

    runningProcesses.forEach(([processId, _]) => {
      this.killProcess(processId, signal);
    });
  }

  /**
   * Clean up finished processes from memory
   * @param {number} maxAge - Maximum age in milliseconds (default: 1 hour)
   */
  cleanup(maxAge = 3600000) {
    const now = Date.now();
    const toDelete = [];

    this.processes.forEach((info, processId) => {
      if (info.status !== 'running' && info.endTime && (now - info.endTime > maxAge)) {
        toDelete.push(processId);
      }
    });

    toDelete.forEach(processId => {
      this.processes.delete(processId);
    });

    if (toDelete.length > 0) {
      console.log(`[ProcessManager] Cleaned up ${toDelete.length} old process(es)`);
    }
  }

  /**
   * Get list of all processes
   * @returns {Array} Array of process info objects
   */
  listProcesses() {
    return Array.from(this.processes.entries()).map(([processId, info]) => ({
      id: processId,
      pid: info.pid,
      command: info.command,
      status: info.status,
      startTime: info.startTime,
      uptime: info.endTime 
        ? info.endTime - info.startTime 
        : Date.now() - info.startTime
    }));
  }
}

// Export singleton instance
module.exports = new ProcessManager();
