/**
 * Alert State Watcher
 * Monitors alert state file with boolean tuple for real-time alerts
 */

const fs = require('fs');
const { EventEmitter } = require('events');

// Alert type definitions mapped to tuple indices
const ALERT_TYPES = {
    0: {
        id: 'cheating_phone_detected',
        severity: 'critical',
        category: 'phone_detected',
        message: ' Phone Detected - Cheating attempt identified' 
    },
    1: {
        id: 'no_face',
        severity: 'warning',
        category: 'no_face',
        message: ' No Face Detected - Please stay in view' 
    },
    2: {
        id: 'multiple_faces',
        severity: 'warning',
        category: 'multiple_faces',
        message: ' Multiple Faces Detected - Only one person allowed'
    },
    3: {
        id: 'face_mismatch',
        severity: 'warning',
        category: 'face_verification',
        message: ' Face Verification Failed - Identity mismatch detected' 
    },
    4: {
        id: 'eye_movement',
        severity: 'warning',
        category: 'eye_movement',
        message: ' Suspicious Eye Movement - Looking away from screen'
    }
};

class AlertStateWatcher extends EventEmitter {
    constructor() {
        super();
        this.alertStateFile = null;
        this.previousState = null;
        this.pollingInterval = null;
        this.pollInterval = 200; // Check every 200ms for faster response
        this.isWatching = false;
    }

    /**
     * Start watching the alert state file
     * @param {string} filePath - Path to alert state file
     * @param {Object} options - Watch options
     */
    startWatching(filePath, options = {}) {
        if (this.isWatching) {
            console.warn('[AlertStateWatcher] Already watching');
            return;
        }

        this.alertStateFile = filePath;
        this.pollInterval = options.pollInterval || 200;
        this.previousState = null;

        // Initialize previous state
        this.readAlertState().then(state => {
            if (state) {
                this.previousState = state;
            }
        });

        // Start polling
        this.pollingInterval = setInterval(() => {
            this.checkForChanges();
        }, this.pollInterval);

        this.isWatching = true;
        console.log(`[AlertStateWatcher] Started watching ${filePath}`);
    }

    /**
     * Stop watching
     */
    stopWatching() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }

        this.isWatching = false;
        this.previousState = null;
        this.alertStateFile = null;

        console.log('[AlertStateWatcher] Stopped watching');
    }

    /**
     * Read alert state from file
     * Expected format: [0,1,0,0,1] or {"alerts": [0,1,0,0,1]}
     * @returns {Promise<Array|null>} Alert state array or null
     */
    async readAlertState() {
        try {
            if (!fs.existsSync(this.alertStateFile)) {
                return null;
            }

            const content = await fs.promises.readFile(this.alertStateFile, 'utf8');
            const trimmed = content.trim();

            if (!trimmed) {
                return null;
            }

            // Try parsing as JSON
            try {
                const parsed = JSON.parse(trimmed);
                
                // Handle both array format [0,1,0,0,1] and object format {"alerts": [0,1,0,0,1]}
                if (Array.isArray(parsed)) {
                    return parsed;
                } else if (parsed.alerts && Array.isArray(parsed.alerts)) {
                    return parsed.alerts;
                }
            } catch (e) {
                // Not JSON, try parsing as comma-separated values
                if (trimmed.includes(',')) {
                    return trimmed.split(',').map(v => parseInt(v.trim()));
                }
            }

            return null;
        } catch (error) {
            if (error.code !== 'ENOENT') {
                console.error('[AlertStateWatcher] Error reading alert state:', error.message);
            }
            return null;
        }
    }

    /**
     * Check for state changes
     */
    async checkForChanges() {
        const currentState = await this.readAlertState();

        if (!currentState) {
            return;
        }

        // Initialize previous state if first read
        if (!this.previousState) {
            this.previousState = currentState;
            return;
        }

        // Check for changes (0 -> 1 transitions)
        const changes = this.detectStateChanges(this.previousState, currentState);

        if (changes.length > 0) {
            // Emit alerts for each change
            for (const change of changes) {
                this.emitAlert(change);
            }

            // Update previous state
            this.previousState = currentState;
        }
    }

    /**
     * Detect state changes (0 -> 1 transitions)
     * @param {Array} oldState - Previous state
     * @param {Array} newState - Current state
     * @returns {Array} Array of change objects
     */
    detectStateChanges(oldState, newState) {
        const changes = [];

        const maxLen = Math.max(oldState.length, newState.length);

        for (let i = 0; i < maxLen; i++) {
            const oldVal = oldState[i] || 0;
            const newVal = newState[i] || 0;

            // Detect 0 -> 1 transition
            if (oldVal === 0 && newVal === 1) {
                const alertDef = ALERT_TYPES[i];
                
                if (alertDef) {
                    changes.push({
                        index: i,
                        alertType: alertDef.id,
                        severity: alertDef.severity,
                        category: alertDef.category,
                        message: alertDef.message,
                        timestamp: new Date().toISOString(),
                        timestampMs: Date.now()
                    });
                }
            }
        }

        return changes;
    }

    /**
     * Emit alert event
     * @param {Object} alert - Alert object
     */
    emitAlert(alert) {
        console.log(`[AlertStateWatcher] Alert triggered: ${alert.alertType} - ${alert.message}`);
        
        // Emit the alert event
        this.emit('alert', alert);

        // Emit notification event (ready for display)
        const notification = {
            type: alert.severity === 'critical' ? 'error' : 'warning',
            message: alert.message,
            category: alert.category,
            timestamp: alert.timestamp
        };

        this.emit('notification', notification);
    }

    /**
     * Get current alert state
     * @returns {Promise<Array|null>} Current state
     */
    async getCurrentState() {
        return await this.readAlertState();
    }

    /**
     * Get alert type definition by index
     * @param {number} index - Alert index
     * @returns {Object|null} Alert definition
     */
    getAlertDefinition(index) {
        return ALERT_TYPES[index] || null;
    }

    /**
     * Get all alert type definitions
     * @returns {Object} All alert definitions
     */
    getAllAlertDefinitions() {
        return { ...ALERT_TYPES };
    }
}

// Export singleton instance
module.exports = new AlertStateWatcher();
