# Quick Reference: Electron App Data Flow

## 1️⃣ Why DevTools Open Automatically?

**File**: [electron/main.js](electron/main.js#L41)

```javascript
if (isDev) {
  mainWindow.loadURL('http://localhost:5173');
  mainWindow.webContents.openDevTools();  // ← THIS LINE
}
```

**Solution for Development**: Comment out line 41
**Solution for Production**: Nothing needed - auto-disabled when you run `npm run build`

---

## 2️⃣ How to Enable Kiosk Mode

**File**: [electron/main.js](electron/main.js#L12-L31)

Add these options to `BrowserWindow`:

```javascript
mainWindow = new BrowserWindow({
  // ... existing options
  fullscreen: true,     // Start fullscreen
  kiosk: true,         // Lock fullscreen (can't exit with F11/Esc)
  alwaysOnTop: true,   // Keep window on top
  // frame: false,     // Remove title bar (optional)
});
```

---

## 3️⃣ Process Spawning Chain

```
React Component (Test Page)
    ↓ button click
useProctoring Hook
    ↓ startMonitoring()
window.electron.invoke('proctoring:start')
    ↓ IPC
Electron Main Process (main.js)
    ↓ ipcMain.handle('proctoring:start')
cameraMonitor.startMonitoring()
    ↓ (camMonitorSpawn.js)
processManager.spawnProcess()
    ↓ (processManager.js)
Python Subprocess spawned
    ↓ child_process.spawn()
HD_ML_stuff/proctor_main_background.py running
```

---

## 4️⃣ Log Access Chain

```
Python writes:
HD_ML_stuff/logs/proctoring/{sessionId}/session_{id}.log
    ↓
LogWatcher polls file every 500ms
    ↓ (logWatcher.js)
New content detected
    ↓
LogParser parses line
    ↓ (logParser.js)
{
  isAlert: true,
  alertType: "Eye Risk",
  alertMessage: "Suspicious eye movement",
  category: "eye_movement"
}
    ↓
Converted to notification format
    ↓
IPC: send('proctoring:notification')
    ↓ (main.js)
Preload script exposes
    ↓ (preload.js)
React receives via window.electron.receive
    ↓ (useProctoring.js)
notify.warning() displays toast
    ↓
🔔 Notification appears on screen
```

---

## 5️⃣ Alert Notification Display

**File**: [src/hooks/useProctoring.js](src/hooks/useProctoring.js#L209-L229)

```javascript
window.electron.receive('proctoring:notification', (notification) => {
  const { type, message, category } = notification.notification;
  
  // Different colors/icons based on type
  if (type === 'error')   notify.error(message, 3000, category);
  if (type === 'warning') notify.warning(message, 3000, category);
  if (type === 'success') notify.success(message, 3000, category);
  if (type === 'info')    notify.info(message, 3000, category);
});
```

**Parameters:**
- `message` - Text to display
- `3000` - Duration in milliseconds (3 seconds)
- `category` - Used for deduplication (prevents spam)

---

## 6️⃣ File Structure

```
frontend/Proctor_App/
├── electron/
│   ├── main.js                    ← Entry point, IPC handlers
│   ├── preload.js                 ← Security bridge
│   └── utils/
│       ├── camMonitorSpawn.js     ← Spawns Python process
│       ├── processManager.js      ← Manages child processes
│       ├── logWatcher.js          ← Watches log files
│       ├── logParser.js           ← Parses log format
│       └── processTerminator.js   ← Stops processes
├── src/
│   └── hooks/
│       └── useProctoring.js       ← React interface
└── HD_ML_stuff/
    ├── proctor_main_background.py ← Python CV models
    └── logs/proctoring/           ← Log files written here
```

---

## 7️⃣ Common Tasks

### Disable DevTools in Development
**File**: `electron/main.js` line 41
```javascript
// mainWindow.webContents.openDevTools();  // Commented out
```

### Change Log Polling Frequency
**File**: `electron/utils/logWatcher.js` line 26
```javascript
pollInterval = 1000,  // Change from 500ms to 1000ms
```

### Modify Notification Duration
**File**: `src/hooks/useProctoring.js` line 220
```javascript
notify.warning(message, 5000, category);  // 5 seconds instead of 3
```

### Add New IPC Channel
1. **Preload** (`electron/preload.js`) - Add to `validChannels`
2. **Main** (`electron/main.js`) - Add `ipcMain.handle()` handler
3. **React** (`src/hooks/useProctoring.js`) - Call `window.electron.invoke()`

---

## 8️⃣ Build Commands

```bash
# Development (DevTools enabled)
npm run dev

# Production Build (DevTools disabled)
npm run build        # All platforms
npm run build:win    # Windows only
npm run build:mac    # macOS only
npm run build:linux  # Linux only
```

---

## 9️⃣ Environment Detection

```javascript
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
```

- **Development**: `NODE_ENV=development` OR app not packaged
- **Production**: After `electron-builder` runs, `app.isPackaged = true`

---

## 🔟 Security Features

✅ Context Isolation - Renderer can't access Node.js directly  
✅ Preload Whitelist - Only specific IPC channels allowed  
✅ Web Security - Enabled in production  
✅ Navigation Prevention - Blocks external URLs  
✅ No Node Integration - Renderer process isolated  

---

## Key Files at a Glance

| What | Where |
|------|-------|
| DevTools toggle | `electron/main.js:41` |
| Spawn Python | `electron/utils/camMonitorSpawn.js:160` |
| Watch logs | `electron/utils/logWatcher.js:24` |
| Parse alerts | `electron/utils/logParser.js:10` |
| Show notifications | `src/hooks/useProctoring.js:209` |
| IPC handlers | `electron/main.js:59-170` |
| Preload bridge | `electron/preload.js:1-48` |

---

**For detailed explanations, see [ARCHITECTURE_FLOW.md](ARCHITECTURE_FLOW.md)**
