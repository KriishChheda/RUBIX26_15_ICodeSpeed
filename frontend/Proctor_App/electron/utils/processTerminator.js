/**
 * Process Termination Utilities
 * Handles graceful and forceful process termination
 */

const processManager = require('./processManager');

class ProcessTerminator {
  constructor() {
    this.cleanupHandlers = new Map();
  }

  /**
   * Register a cleanup handler for a specific process
   * @param {number} processId - Process ID
   * @param {Function} handler - Cleanup function to run before termination
   */
  registerCleanupHandler(processId, handler) {
    this.cleanupHandlers.set(processId, handler);
  }

  /**
   * Gracefully terminate a process
   * Sends SIGTERM, waits for graceful exit, then SIGKILL if needed
   * @param {number} processId - Process ID to terminate
   * @param {Object} options - Termination options
   * @returns {Promise<Object>} Result object
   */
  async gracefulTerminate(processId, options = {}) {
    const {
      timeout = 5000,      // Wait 5 seconds before force kill
      runCleanup = true    // Run cleanup handler
    } = options;

    console.log(`[ProcessTerminator] Gracefully terminating process ${processId}`);

    // Check if process exists and is running
    if (!processManager.isRunning(processId)) {
      console.warn(`[ProcessTerminator] Process ${processId} is not running`);
      return {
        success: false,
        error: 'Process not running',
        processId
      };
    }

    // Run cleanup handler if registered
    if (runCleanup && this.cleanupHandlers.has(processId)) {
      try {
        console.log(`[ProcessTerminator] Running cleanup handler for process ${processId}`);
        const handler = this.cleanupHandlers.get(processId);
        await handler();
      } catch (error) {
        console.error(`[ProcessTerminator] Cleanup handler error: ${error.message}`);
      }
    }

    // Send SIGTERM
    const terminated = processManager.killProcess(processId, 'SIGTERM');
    
    if (!terminated) {
      return {
        success: false,
        error: 'Failed to send SIGTERM',
        processId
      };
    }

    // Wait for graceful exit
    const exited = await this.waitForExit(processId, timeout);

    if (exited) {
      console.log(`[ProcessTerminator] Process ${processId} exited gracefully`);
      this.cleanupHandlers.delete(processId);
      return {
        success: true,
        method: 'graceful',
        processId
      };
    }

    // Force kill if still running
    console.warn(`[ProcessTerminator] Process ${processId} did not exit gracefully, force killing`);
    const killed = processManager.killProcess(processId, 'SIGKILL');
    
    if (killed) {
      // Wait a bit for SIGKILL to take effect
      await this.delay(500);
      this.cleanupHandlers.delete(processId);
      
      return {
        success: true,
        method: 'forced',
        processId
      };
    }

    return {
      success: false,
      error: 'Failed to kill process',
      processId
    };
  }

  /**
   * Force terminate a process immediately
   * Sends SIGKILL without waiting
   * @param {number} processId - Process ID to terminate
   * @returns {Object} Result object
   */
  forceTerminate(processId) {
    console.log(`[ProcessTerminator] Force terminating process ${processId}`);

    const killed = processManager.killProcess(processId, 'SIGKILL');
    this.cleanupHandlers.delete(processId);

    return {
      success: killed,
      method: 'forced',
      processId,
      error: killed ? null : 'Failed to kill process'
    };
  }

  /**
   * Terminate all running processes
   * @param {Object} options - Termination options
   * @returns {Promise<Object>} Results object
   */
  async terminateAll(options = {}) {
    const {
      graceful = true,
      timeout = 5000
    } = options;

    console.log('[ProcessTerminator] Terminating all processes');

    const processes = processManager.listProcesses()
      .filter(p => p.status === 'running');

    if (processes.length === 0) {
      console.log('[ProcessTerminator] No running processes to terminate');
      return {
        success: true,
        terminated: 0
      };
    }

    const results = [];

    if (graceful) {
      // Terminate gracefully in parallel
      const promises = processes.map(p => 
        this.gracefulTerminate(p.id, { timeout, runCleanup: true })
      );
      
      const settled = await Promise.allSettled(promises);
      settled.forEach((result, index) => {
        results.push({
          processId: processes[index].id,
          success: result.status === 'fulfilled' && result.value.success,
          method: result.status === 'fulfilled' ? result.value.method : 'error',
          error: result.status === 'rejected' ? result.reason : null
        });
      });
    } else {
      // Force terminate all
      processes.forEach(p => {
        const result = this.forceTerminate(p.id);
        results.push(result);
      });
    }

    const successful = results.filter(r => r.success).length;

    console.log(`[ProcessTerminator] Terminated ${successful}/${processes.length} processes`);

    return {
      success: successful === processes.length,
      total: processes.length,
      terminated: successful,
      results
    };
  }

  /**
   * Wait for a process to exit
   * @param {number} processId - Process ID
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<boolean>} True if exited within timeout
   */
  async waitForExit(processId, timeout) {
    const startTime = Date.now();
    const pollInterval = 100; // Check every 100ms

    while (Date.now() - startTime < timeout) {
      if (!processManager.isRunning(processId)) {
        return true;
      }
      await this.delay(pollInterval);
    }

    return false;
  }

  /**
   * Delay helper
   * @param {number} ms - Milliseconds to delay
   * @returns {Promise} Promise that resolves after delay
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Terminate process by PID (system process ID)
   * Useful for orphaned processes
   * @param {number} pid - System process ID
   * @param {string} signal - Signal to send
   * @returns {boolean} Success status
   */
  terminateByPid(pid, signal = 'SIGTERM') {
    try {
      process.kill(pid, signal);
      console.log(`[ProcessTerminator] Sent ${signal} to PID ${pid}`);
      return true;
    } catch (error) {
      console.error(`[ProcessTerminator] Failed to kill PID ${pid}: ${error.message}`);
      return false;
    }
  }

  /**
   * Clean up terminated processes and handlers
   */
  cleanup() {
    // Clean up old processes from ProcessManager
    processManager.cleanup();

    // Clean up orphaned cleanup handlers
    const runningProcessIds = new Set(
      processManager.listProcesses()
        .filter(p => p.status === 'running')
        .map(p => p.id)
    );

    this.cleanupHandlers.forEach((_, processId) => {
      if (!runningProcessIds.has(processId)) {
        this.cleanupHandlers.delete(processId);
      }
    });

    console.log('[ProcessTerminator] Cleanup complete');
  }

  /**
   * Get terminator status
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      runningProcesses: processManager.listProcesses().filter(p => p.status === 'running').length,
      registeredHandlers: this.cleanupHandlers.size,
      processes: processManager.listProcesses()
    };
  }
}

// Export singleton instance
module.exports = new ProcessTerminator();
