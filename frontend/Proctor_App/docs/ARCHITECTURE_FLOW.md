# Electron Proctor App - Architecture & Data Flow

## Overview
This document explains how the Electron app spawns the background monitoring process, accesses log files, and displays notifications.

---

## 1. DevTools Configuration

### Development Mode
- **File**: [electron/main.js](electron/main.js#L38-L45)
- DevTools open automatically when `NODE_ENV=development` or app is not packaged
- To disable: Comment out `mainWindow.webContents.openDevTools()` on line 41

### Production Mode
- **Build Command**: `npm run build` or `npm run build:win`
- DevTools are **automatically disabled** in production builds
- The app checks `app.isPackaged` to determine the environment
- No additional configuration needed for production security

### Kiosk-Like Mode (Production)
To enforce a locked-down experience in production builds, the app already:
- ✅ Disables DevTools (`!isDev` check)
- ✅ Prevents external navigation ([main.js](electron/main.js#L232-L248))
- ✅ Enforces web security
- ✅ Uses context isolation

To make it more restrictive (optional):
```javascript
// In createWindow() function
fullscreen: true,          // Start in fullscreen
kiosk: true,              // Enable kiosk mode (locks fullscreen)
alwaysOnTop: true,        // Keep window on top
frame: false,             // Remove window frame
```

---

## 2. Background Monitor Process Flow

### Process Spawning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         React Frontend                           │
│  (src/hooks/useProctoring.js)                                   │
│  User clicks "Start Monitoring"                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ IPC: invoke('proctoring:start')
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Electron Main Process                         │
│  (electron/main.js)                                             │
│  IPC Handler: ipcMain.handle('proctoring:start')               │
└────────────────────────┬────────────────────────────────────────┘
                         │ calls cameraMonitor.startMonitoring()
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               Camera Monitor Spawner                             │
│  (electron/utils/camMonitorSpawn.js)                           │
│  - Finds Python executable                                      │
│  - Validates paths (Python, script, participant image)         │
│  - Builds command: python proctor_main_background.py           │
└────────────────────────┬────────────────────────────────────────┘
                         │ calls processManager.spawnProcess()
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Process Manager                                │
│  (electron/utils/processManager.js)                            │
│  - Spawns Python subprocess using Node.js child_process        │
│  - Captures stdout/stderr                                       │
│  - Tracks process by unique ID                                  │
│  - Emits events: onOutput, onExit, onError                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            Python Background Process                             │
│  (HD_ML_stuff/proctor_main_background.py)                      │
│  - Runs CV models (face detection, eye tracking, etc.)         │
│  - Writes logs to HD_ML_stuff/logs/proctoring/{sessionId}/    │
│  - Generates alerts in log files                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

1. **[electron/utils/camMonitorSpawn.js](electron/utils/camMonitorSpawn.js)** - Main spawner
   - `initialize()` - Finds Python path and validates environment
   - `startMonitoring()` - Spawns the Python background process
   - `stopMonitoring()` - Gracefully terminates the process

2. **[electron/utils/processManager.js](electron/utils/processManager.js)** - Process lifecycle
   - Manages multiple child processes
   - Captures and buffers stdout/stderr
   - Provides process status and logs

3. **[electron/main.js](electron/main.js#L59-L115)** - IPC bridge
   - Handles `proctoring:start` - Starts background process
   - Handles `proctoring:stop` - Stops background process
   - Handles `proctoring:status` - Gets current status

---

## 3. Log File Access & Monitoring

### How Frontend Accesses Logs

```
┌─────────────────────────────────────────────────────────────────┐
│            Python Background Process                             │
│  Writes logs to:                                                │
│  HD_ML_stuff/logs/proctoring/{sessionId}/session_{id}.log      │
└────────────────────────┬────────────────────────────────────────┘
                         │ File written continuously
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Log File Watcher                              │
│  (electron/utils/logWatcher.js)                                 │
│  - Polls log file every 500ms                                   │
│  - Tracks file position (reads only new content)               │
│  - Emits events: 'raw', 'line', 'alert', 'notification'       │
└────────────────────────┬────────────────────────────────────────┘
                         │ Event: 'alert' or 'notification'
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Log Parser                                   │
│  (electron/utils/logParser.js)                                  │
│  - Parses log line format: "2026-01-21 02:29:07 - INFO - ..."  │
│  - Extracts alert format: "[WARNING] Type: Message"            │
│  - Categorizes alerts: face, eye, phone, etc.                  │
│  - Converts to notification format                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Camera Monitor Spawner                              │
│  (electron/utils/camMonitorSpawn.js)                           │
│  Forwards events via callbacks:                                 │
│  - onLogAlert(alert)                                            │
│  - onLogNotification(notification)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ IPC: send('proctoring:notification')
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Electron Main Process                           │
│  (electron/main.js)                                             │
│  Forwards to renderer:                                          │
│  mainWindow.webContents.send('proctoring:notification', ...)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Preload Script                                │
│  (electron/preload.js)                                          │
│  Exposes IPC via contextBridge:                                 │
│  window.electron.receive('proctoring:notification', callback)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   React Frontend                                 │
│  (src/hooks/useProctoring.js)                                   │
│  useEffect(() => {                                              │
│    window.electron.receive('proctoring:notification', ...)     │
│  })                                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Log File Watcher Details

**[electron/utils/logWatcher.js](electron/utils/logWatcher.js)**

1. **Polling Mechanism**
   - Uses `setInterval()` to check file every 500ms
   - Tracks last read position using `fs.statSync()`
   - Only reads new bytes: `currentSize - lastPosition`

2. **Event Types**
   - `raw` - Raw new content from file
   - `line` - Parsed individual log line
   - `alert` - Parsed alert with severity
   - `notification` - Ready-to-display notification

3. **Why Not fs.watch()?**
   - More reliable across platforms
   - Avoids multiple events for single write
   - Better control over reading frequency

---

## 4. Alert Notification Flow

### Complete Flow from Python to UI

```
Python logs:
"2026-01-21 02:29:19 - WARNING - [WARNING] Eye Risk: Suspicious eye movement"
                         │
                         ▼
LogWatcher reads new line, sends to LogParser
                         │
                         ▼
LogParser.parseLogLine() returns:
{
  timestamp: "2026-01-21 02:29:19",
  level: "warning",
  isAlert: true,
  alertType: "Eye Risk",
  alertMessage: "Suspicious eye movement",
  severity: "warning",
  category: "eye_movement"
}
                         │
                         ▼
LogParser.logToNotification() converts to:
{
  notification: {
    type: "warning",
    message: "⚠️ Eye Risk: Suspicious eye movement",
    category: "eye_movement"
  }
}
                         │
                         ▼
Main process sends IPC:
mainWindow.webContents.send('proctoring:notification', notification)
                         │
                         ▼
Frontend receives via preload script:
window.electron.receive('proctoring:notification', callback)
                         │
                         ▼
React hook processes notification:
(src/hooks/useProctoring.js)
useEffect(() => {
  window.electron.receive('proctoring:notification', (notification) => {
    notify.warning(message, 3000, category);  // Shows toast
  });
})
                         │
                         ▼
Notification appears on screen (via notify.warning())
```

### Notification System

**[src/hooks/useProctoring.js](frontend/Proctor_App/src/hooks/useProctoring.js#L209-L229)** - Displays notifications

```javascript
window.electron.receive('proctoring:notification', (notification) => {
  const { type, message, category } = notification.notification;
  
  // Display based on severity
  if (type === 'error') {
    notify.error(message, 3000, category);
  } else if (type === 'warning') {
    notify.warning(message, 3000, category);
  } else if (type === 'success') {
    notify.success(message, 3000, category);
  } else {
    notify.info(message, 3000, category);
  }
});
```

**Key Features:**
- Uses `category` for deduplication (prevents spam)
- Different visual styles based on `type` (error, warning, success, info)
- Auto-dismiss after 3000ms (3 seconds)
- Toast notifications appear at top-right (or wherever configured)

---

## 5. Security & IPC Bridge

### Electron Security Model

**[electron/preload.js](electron/preload.js)** - Context Bridge
- Uses `contextBridge` to safely expose limited IPC methods
- Only whitelisted channels are allowed
- Renderer process cannot directly access Node.js APIs

**Whitelisted Channels:**
```javascript
validChannels = [
  'proctoring:start',
  'proctoring:stop', 
  'proctoring:status',
  'proctoring:getLogs',
  'proctoring:setParticipantImage',
  'proctoring:getAlertsJSON',
  'proctoring:output',
  'proctoring:stopped',
  'proctoring:alert',
  'proctoring:notification'
]
```

---

## 6. Production Build Configuration

### Current Settings

**[package.json](package.json)** - Build scripts
```json
{
  "scripts": {
    "dev": "concurrently \"vite\" \"wait-on http://localhost:5173 && cross-env NODE_ENV=development electron .\"",
    "build": "vite build && electron-builder",
    "build:win": "vite build && electron-builder --win"
  }
}
```

**[electron-builder.json](electron-builder.json)** - Packaging config
- Defines what files to include
- Sets app name, version, icons
- Configures installers for different platforms

### What Happens in Production Build

1. **Vite Build**
   - Frontend React app is bundled to `dist/` folder
   - All assets are optimized and minified

2. **Electron Builder**
   - Packages the app with Electron runtime
   - Includes `electron/` folder and `dist/` folder
   - `app.isPackaged` becomes `true`
   - DevTools are automatically disabled (checked via `isDev` flag)

3. **File Access**
   - Python scripts must be included in build
   - App root is `path.dirname(app.getPath('exe'))` in production
   - Logs are written relative to executable location

---

## 7. Key Takeaways

### ✅ DevTools Issue
- **Cause**: Line 41 in [electron/main.js](electron/main.js#L41) calls `openDevTools()` in dev mode
- **Fix Applied**: Added comment to explain it's development-only
- **Production**: DevTools are automatically disabled (no code change needed)
- **To Disable in Dev**: Comment out line 41

### ✅ Background Process Spawning
- **Where**: [electron/utils/camMonitorSpawn.js](electron/utils/camMonitorSpawn.js) spawns Python process
- **Triggered by**: IPC call from React frontend via `useProctoring` hook
- **Process Manager**: [electron/utils/processManager.js](electron/utils/processManager.js) handles lifecycle

### ✅ Log File Access
- **Method**: Polling-based file watcher (not fs.watch)
- **File**: [electron/utils/logWatcher.js](electron/utils/logWatcher.js)
- **Frequency**: Checks every 500ms for new content
- **Parser**: [electron/utils/logParser.js](electron/utils/logParser.js) extracts alerts

### ✅ Notifications
- **Flow**: Python log → LogWatcher → LogParser → IPC → Preload → React → Toast
- **Display**: [src/hooks/useProctoring.js](frontend/Proctor_App/src/hooks/useProctoring.js#L209-L229) uses `notify` hook
- **Deduplication**: Uses `category` to prevent alert spam

---

## 8. File Reference Index

| Component | File Path | Purpose |
|-----------|-----------|---------|
| Main Electron Entry | `electron/main.js` | Creates window, handles IPC, manages lifecycle |
| Preload Script | `electron/preload.js` | Security bridge between main and renderer |
| Camera Monitor Spawner | `electron/utils/camMonitorSpawn.js` | Spawns Python background process |
| Process Manager | `electron/utils/processManager.js` | Manages child processes |
| Log Watcher | `electron/utils/logWatcher.js` | Monitors log files for changes |
| Log Parser | `electron/utils/logParser.js` | Parses log format and alerts |
| Process Terminator | `electron/utils/processTerminator.js` | Gracefully stops processes |
| React Hook | `src/hooks/useProctoring.js` | Frontend interface to proctoring system |
| Python Script | `HD_ML_stuff/proctor_main_background.py` | CV models and monitoring logic |

---

## Need Help?

- **DevTools in Dev**: Comment line 41 in [electron/main.js](electron/main.js)
- **Kiosk Mode**: Add `kiosk: true` to `BrowserWindow` options
- **Notification Styling**: Modify `useNotification` hook
- **Log Format**: Edit [logParser.js](electron/utils/logParser.js)
- **Process Management**: See [processManager.js](electron/utils/processManager.js)
