# Alert State File Communication System

## Overview
This system uses a tuple-based file watching approach for real-time alert notifications. The Python process writes a boolean array to a file, and the frontend monitors state changes to trigger notifications.

---

## Alert State File

### Location
```
HD_ML_stuff/logs/proctoring/{sessionId}/alert_state.txt
```

### Format
The file contains a JSON array of boolean values (0 or 1):

```json
[0,1,0,0,1]
```

Or optionally as an object:
```json
{"alerts": [0,1,0,0,1]}
```

### Tuple Index Mapping

| Index | Alert Type | Severity | Message |
|-------|------------|----------|---------|
| 0 | `cheating_phone_detected` | CRITICAL | 📱 Phone Detected - Cheating attempt identified |
| 1 | `no_face` | WARNING | 👤 No Face Detected - Please stay in view |
| 2 | `multiple_faces` | WARNING | 👥 Multiple Faces Detected - Only one person allowed |
| 3 | `face_mismatch` | WARNING | ⚠️ Face Verification Failed - Identity mismatch detected |
| 4 | `eye_movement` | WARNING | 👁️ Suspicious Eye Movement - Looking away from screen |

---

## How It Works

### Python Side (HD_ML_stuff)
Your Python proctoring script should:

1. **Initialize alert state** (at session start):
```python
alert_state = [0, 0, 0, 0, 0]  # All alerts off
```

2. **Update state when alert triggered**:
```python
# Example: Phone detected
alert_state[0] = 1  # Set phone_detected alert

# Example: No face detected
alert_state[1] = 1  # Set no_face alert

# Example: Clear an alert
alert_state[1] = 0  # Clear no_face alert
```

3. **Write to file** (continuously or on change):
```python
import json
import os

# Get session log directory
session_id = os.environ.get('PROCTOR_SESSION_ID')
log_dir = f'logs/proctoring/{session_id}'
alert_file = os.path.join(log_dir, 'alert_state.txt')

# Write current state
with open(alert_file, 'w') as f:
    json.dump(alert_state, f)
```

### Electron Side (Frontend)

**File**: [electron/utils/alertStateWatcher.js](electron/utils/alertStateWatcher.js)

1. **Polls the file** every 200ms
2. **Reads current state** as JSON array
3. **Compares with previous state**
4. **Detects 0→1 transitions** (alert triggered)
5. **Emits alert event** with fixed message
6. **IPC sends to React frontend**

---

## Frontend Display

**File**: [src/hooks/useProctoring.js](src/hooks/useProctoring.js#L203-L220)

Alerts are received and displayed automatically:

```javascript
window.electron.receive('proctoring:alert', (alert) => {
  const { severity, message, category } = alert;
  
  if (severity === 'critical') {
    notify.error(message, 5000, category);  // Red, 5 seconds
  } else {
    notify.warning(message, 3000, category); // Orange, 3 seconds
  }
});
```

---

## Notification Features

### Deduplication
- Uses `category` to prevent spam
- Same alert won't show multiple notifications simultaneously
- New alert of same type replaces previous one

### Display Duration
- **Critical alerts** (phone detected): 5 seconds
- **Warning alerts** (face/eye issues): 3 seconds

### Automatic Clearing
- When state changes from `1→0`, the notification naturally expires
- No need to manually clear notifications

---

## Python Implementation Example

```python
import json
import os
import time

class AlertStateManager:
    def __init__(self, session_id):
        self.session_id = session_id
        self.log_dir = f'logs/proctoring/{session_id}'
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.alert_file = os.path.join(self.log_dir, 'alert_state.txt')
        
        # Initialize all alerts as off
        self.alert_state = [0, 0, 0, 0, 0]
        self.write_state()
    
    def set_alert(self, index, value):
        """Set alert at index to value (0 or 1)"""
        if 0 <= index < len(self.alert_state):
            old_value = self.alert_state[index]
            self.alert_state[index] = value
            
            # Only write if state changed
            if old_value != value:
                self.write_state()
    
    def write_state(self):
        """Write current alert state to file"""
        try:
            with open(self.alert_file, 'w') as f:
                json.dump(self.alert_state, f)
        except Exception as e:
            print(f"Error writing alert state: {e}")
    
    # Convenience methods
    def trigger_phone_detected(self):
        self.set_alert(0, 1)
    
    def clear_phone_detected(self):
        self.set_alert(0, 0)
    
    def trigger_no_face(self):
        self.set_alert(1, 1)
    
    def clear_no_face(self):
        self.set_alert(1, 0)
    
    def trigger_multiple_faces(self):
        self.set_alert(2, 1)
    
    def clear_multiple_faces(self):
        self.set_alert(2, 0)
    
    def trigger_face_mismatch(self):
        self.set_alert(3, 1)
    
    def clear_face_mismatch(self):
        self.set_alert(3, 0)
    
    def trigger_eye_movement(self):
        self.set_alert(4, 1)
    
    def clear_eye_movement(self):
        self.set_alert(4, 0)

# Usage in your proctoring script
session_id = os.environ.get('PROCTOR_SESSION_ID')
alert_manager = AlertStateManager(session_id)

# In your detection loop
while monitoring:
    # ... your CV detection code ...
    
    if phone_detected:
        alert_manager.trigger_phone_detected()
    else:
        alert_manager.clear_phone_detected()
    
    if no_face_detected:
        alert_manager.trigger_no_face()
    else:
        alert_manager.clear_no_face()
    
    # ... etc for other alerts ...
```

---

## Configuration

### Change Poll Interval
**File**: [electron/utils/alertStateWatcher.js](electron/utils/alertStateWatcher.js#L43)

```javascript
this.pollInterval = 200; // Change to 100 for faster, 500 for slower
```

Or in [electron/utils/camMonitorSpawn.js](electron/utils/camMonitorSpawn.js#L229):
```javascript
alertStateWatcher.startWatching(alertStateFile, { pollInterval: 100 });
```

### Add New Alert Type
1. Add to `ALERT_TYPES` in [alertStateWatcher.js](electron/utils/alertStateWatcher.js#L8-L36)
2. Update your Python array to include the new index
3. Messages are automatically displayed

### Modify Alert Messages
Edit the `message` field in `ALERT_TYPES` object:

```javascript
const ALERT_TYPES = {
  0: {
    id: 'cheating_phone_detected',
    severity: 'critical',
    category: 'phone_detected',
    message: '📱 Your custom message here'  // ← Change this
  },
  // ... etc
};
```

---

## Advantages of This Approach

✅ **Simple**: Just write a JSON array, no complex log parsing  
✅ **Fast**: 200ms polling for near-instant notifications  
✅ **Fixed Messages**: Consistent, localized messages in frontend  
✅ **State-Based**: Automatically handles alert clearing  
✅ **Deduplication**: Built-in spam prevention  
✅ **Separation**: Alerts are separate from logs  
✅ **Extensible**: Easy to add new alert types  

---

## File Monitoring

### Separate File vs Same Log File?

**✅ Using Separate File** (Current Implementation)
- **Alert State File**: `alert_state.txt` - Real-time boolean state
- **Log File**: `session_{id}.log` - Historical records (optional)

**Benefits**:
- Faster parsing (simple JSON vs complex log format)
- Clear separation of concerns
- Alert state is always current
- No risk of parsing errors from malformed log lines
- Can disable log watching for better performance

---

## API Reference

### Alert Object Structure

When `proctoring:alert` is received in React:

```javascript
{
  index: 0,                              // Position in tuple
  alertType: 'cheating_phone_detected',  // Alert ID
  severity: 'critical',                  // 'critical' or 'warning'
  category: 'phone_detected',            // For deduplication
  message: '📱 Phone Detected - ...',    // Display message
  timestamp: '2026-01-21T12:34:56.789Z', // ISO timestamp
  timestampMs: 1737462896789             // Unix timestamp
}
```

---

## Troubleshooting

### Alerts Not Showing

1. **Check file exists**:
   - `HD_ML_stuff/logs/proctoring/{sessionId}/alert_state.txt`

2. **Check file format**:
   - Must be valid JSON: `[0,1,0,0,1]`
   - Or: `{"alerts": [0,1,0,0,1]}`

3. **Check console logs**:
   - Look for `[AlertStateWatcher]` messages
   - Check for parsing errors

4. **Verify Python writes**:
   - Add logging after file write
   - Check file permissions

### Duplicate Notifications

- Already handled by `category` deduplication
- Same category notifications replace previous ones

### Delayed Notifications

- Reduce `pollInterval` to 100ms or 50ms
- Default is 200ms (very responsive)

---

## Migration from Old System

If you were using log file parsing before:

**Old System** (Line-based log parsing):
```
"2026-01-21 02:29:19 - WARNING - [WARNING] Eye Risk: Suspicious eye movement"
```

**New System** (State-based):
```json
[0,0,0,0,1]  
```

Both systems can coexist:
- Set `watchLogs: false` to disable old system
- Set `watchAlerts: true` to enable new system (default)

---

**See also**: [ARCHITECTURE_FLOW.md](ARCHITECTURE_FLOW.md) for complete system architecture
