/**
 * Log Parser Utility
 * Parses proctoring log files and extracts structured data
 */

/**
 * Parse a single log line
 * Format: "2026-01-21 02:29:07 - INFO - Message"
 * Alert Format: "2026-01-21 02:29:19 - WARNING - [WARNING] Eye Risk: Suspicious eye movement..."
 * 
 * @param {string} line - Log line to parse
 * @returns {Object|null} Parsed log entry or null if invalid
 */
function parseLogLine(line) {
  if (!line || line.trim() === '') {
    return null;
  }

  // Regex to parse log format: timestamp - level - message
  const logPattern = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) - (.+)$/;
  const match = line.match(logPattern);

  if (!match) {
    return null;
  }

  const [, timestamp, level, message] = match;

  // Parse alert messages with format: [SEVERITY] Type: Message
  const alertPattern = /^\[(\w+)\] ([^:]+): (.+)$/;
  const alertMatch = message.match(alertPattern);

  let parsedEntry = {
    timestamp,
    timestampMs: new Date(timestamp).getTime(),
    level: level.toLowerCase(),
    rawMessage: message,
    isAlert: false,
    alertType: null,
    alertMessage: null,
    severity: level.toLowerCase(),
    metadata: {}
  };

  if (alertMatch) {
    const [, severity, type, alertMessage] = alertMatch;
    parsedEntry.isAlert = true;
    parsedEntry.alertType = type.trim();
    parsedEntry.alertMessage = alertMessage.trim();
    parsedEntry.severity = severity.toLowerCase();
  }

  // Extract metadata from message if present
  const metadataPattern = /\|\s*(\{.+\})$/;
  const metadataMatch = message.match(metadataPattern);
  if (metadataMatch) {
    try {
      parsedEntry.metadata = JSON.parse(metadataMatch[1]);
    } catch (e) {
      // Invalid JSON, ignore
    }
  }

  // Categorize log type
  parsedEntry.category = categorizeLog(parsedEntry);

  return parsedEntry;
}

/**
 * Categorize log entry based on content
 * @param {Object} entry - Parsed log entry
 * @returns {string} Category
 */
function categorizeLog(entry) {
  if (!entry.isAlert) {
    if (entry.rawMessage.includes('SESSION STARTED')) return 'session_start';
    if (entry.rawMessage.includes('SESSION ENDED')) return 'session_end';
    if (entry.rawMessage.includes('Duration:')) return 'session_info';
    if (entry.rawMessage.includes('Total Frames:')) return 'session_info';
    if (entry.rawMessage.includes('Total Alerts:')) return 'session_info';
    return 'info';
  }

  const type = entry.alertType?.toLowerCase() || '';
  
  if (type.includes('face')) {
    if (type.includes('multiple')) return 'multiple_faces';
    if (type.includes('no')) return 'no_face';
    if (type.includes('verification')) return 'face_verification';
    return 'face_detection';
  }
  
  if (type.includes('eye')) {
    return 'eye_movement';
  }
  
  if (type.includes('phone')) {
    return 'phone_detected';
  }
  
  if (type.includes('error')) {
    return 'error';
  }

  return 'unknown';
}

/**
 * Parse entire log file
 * @param {string} logContent - Content of log file
 * @returns {Array<Object>} Array of parsed log entries
 */
function parseLogFile(logContent) {
  if (!logContent) {
    return [];
  }

  const lines = logContent.split('\n');
  const entries = [];

  for (const line of lines) {
    const parsed = parseLogLine(line);
    if (parsed) {
      entries.push(parsed);
    }
  }

  return entries;
}

/**
 * Parse alerts JSON file
 * @param {string} jsonContent - JSON content
 * @returns {Object} Parsed alerts data
 */
function parseAlertsJSON(jsonContent) {
  try {
    const data = JSON.parse(jsonContent);
    
    return {
      sessionId: data.session_id,
      startTime: data.start_time,
      endTime: data.end_time,
      alerts: data.alerts.map(alert => ({
        timestamp: alert.timestamp,
        timestampMs: new Date(alert.timestamp).getTime(),
        type: alert.type,
        message: alert.message,
        severity: alert.severity,
        metadata: alert.metadata || {},
        category: categorizeAlertType(alert.type)
      })),
      statistics: data.statistics
    };
  } catch (error) {
    console.error('Failed to parse alerts JSON:', error);
    return null;
  }
}

/**
 * Categorize alert type
 * @param {string} alertType - Alert type from JSON
 * @returns {string} Category
 */
function categorizeAlertType(alertType) {
  const type = alertType.toLowerCase();
  
  if (type.includes('multiple') && type.includes('face')) return 'multiple_faces';
  if (type.includes('no') && type.includes('face')) return 'no_face';
  if (type.includes('eye')) return 'eye_movement';
  if (type.includes('phone')) return 'phone_detected';
  if (type.includes('verification')) return 'face_verification';
  if (type.includes('error')) return 'error';
  
  return 'unknown';
}

/**
 * Filter logs by criteria
 * @param {Array<Object>} logs - Parsed log entries
 * @param {Object} filters - Filter criteria
 * @returns {Array<Object>} Filtered logs
 */
function filterLogs(logs, filters = {}) {
  let filtered = [...logs];

  if (filters.level) {
    filtered = filtered.filter(log => log.level === filters.level);
  }

  if (filters.severity) {
    filtered = filtered.filter(log => log.severity === filters.severity);
  }

  if (filters.category) {
    filtered = filtered.filter(log => log.category === filters.category);
  }

  if (filters.isAlert !== undefined) {
    filtered = filtered.filter(log => log.isAlert === filters.isAlert);
  }

  if (filters.startTime) {
    const startMs = new Date(filters.startTime).getTime();
    filtered = filtered.filter(log => log.timestampMs >= startMs);
  }

  if (filters.endTime) {
    const endMs = new Date(filters.endTime).getTime();
    filtered = filtered.filter(log => log.timestampMs <= endMs);
  }

  if (filters.search) {
    const searchLower = filters.search.toLowerCase();
    filtered = filtered.filter(log => 
      log.rawMessage.toLowerCase().includes(searchLower) ||
      (log.alertMessage && log.alertMessage.toLowerCase().includes(searchLower))
    );
  }

  return filtered;
}

/**
 * Get alert summary from logs
 * @param {Array<Object>} logs - Parsed log entries
 * @returns {Object} Summary statistics
 */
function getAlertSummary(logs) {
  const alerts = logs.filter(log => log.isAlert);
  
  const summary = {
    totalAlerts: alerts.length,
    bySeverity: {
      info: 0,
      warning: 0,
      critical: 0
    },
    byCategory: {},
    byType: {},
    timeline: []
  };

  alerts.forEach(alert => {
    // Count by severity
    if (summary.bySeverity[alert.severity] !== undefined) {
      summary.bySeverity[alert.severity]++;
    }

    // Count by category
    if (!summary.byCategory[alert.category]) {
      summary.byCategory[alert.category] = 0;
    }
    summary.byCategory[alert.category]++;

    // Count by type
    if (alert.alertType) {
      if (!summary.byType[alert.alertType]) {
        summary.byType[alert.alertType] = 0;
      }
      summary.byType[alert.alertType]++;
    }

    // Add to timeline
    summary.timeline.push({
      timestamp: alert.timestamp,
      timestampMs: alert.timestampMs,
      type: alert.alertType,
      category: alert.category,
      severity: alert.severity
    });
  });

  // Sort timeline
  summary.timeline.sort((a, b) => a.timestampMs - b.timestampMs);

  return summary;
}

/**
 * Convert log entry to notification format
 * @param {Object} logEntry - Parsed log entry
 * @returns {Object|null} Notification object or null if not worthy
 */
function logToNotification(logEntry) {
  // Only create notifications for alerts
  if (!logEntry.isAlert) {
    return null;
  }

  // Determine notification type based on severity
  let notificationType = 'info';
  if (logEntry.severity === 'critical' || logEntry.severity === 'error') {
    notificationType = 'error';
  } else if (logEntry.severity === 'warning') {
    notificationType = 'warning';
  }

  // Create notification message
  let message = logEntry.alertMessage || logEntry.rawMessage;
  
  // Add context based on category
  const categoryPrefixes = {
    'multiple_faces': '👥 ',
    'no_face': '❌ ',
    'eye_movement': '👁️ ',
    'phone_detected': '📱 ',
    'face_verification': '🔒 ',
    'error': '⚠️ '
  };

  const prefix = categoryPrefixes[logEntry.category] || '';
  message = prefix + message;

  return {
    type: notificationType,
    message,
    timestamp: logEntry.timestampMs,
    category: logEntry.category,
    severity: logEntry.severity,
    metadata: logEntry.metadata
  };
}

module.exports = {
  parseLogLine,
  parseLogFile,
  parseAlertsJSON,
  filterLogs,
  getAlertSummary,
  logToNotification,
  categorizeLog,
  categorizeAlertType
};
