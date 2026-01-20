const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const cameraMonitor = require('./utils/camMonitorSpawn');
const processTerminator = require('./utils/processTerminator');
const logWatcher = require('./utils/logWatcher');

// Determine if running in development or production
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      // Enable web security
      webSecurity: true,
      // Allow media access for webcam
      enableWebSQL: false,
    },
    titleBarStyle: 'default',
    backgroundColor: '#0f172a',
    show: false, // Don't show until ready
    icon: path.join(__dirname, '../public/vite.svg')
  });

  // Show window when ready to avoid visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Load the app
  if (isDev) {
    // In development, load from Vite dev server
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built files
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle any unhandled errors
  mainWindow.webContents.on('crashed', () => {
    console.error('Window crashed');
  });
}

// IPC handlers - add your custom handlers here
ipcMain.handle('getVersion', () => {
  return app.getVersion();
});

// Proctoring IPC handlers
ipcMain.handle('proctoring:start', async (event, options) => {
  try {
    // Initialize camera monitor if not already done
    const validation = cameraMonitor.initialize(options.config);
    
    if (!validation.success) {
      return {
        success: false,
        error: 'Failed to initialize camera monitor',
        details: validation.checks
      };
    }

    // Start monitoring
    const result = cameraMonitor.startMonitoring({
      sessionId: options.sessionId,
      faceDetect: options.faceDetect,
      faceMatch: options.faceMatch,
      eyeTracking: options.eyeTracking,
      phoneDetect: options.phoneDetect,
      watchLogs: true,
      onOutput: (data, type) => {
        // Send output to renderer if needed
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('proctoring:output', { data, type });
        }
      },
      onLogAlert: (alert) => {
        // Send parsed alert to renderer
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('proctoring:alert', alert);
        }
      },
      onLogNotification: (notification) => {
        // Send notification-ready data to renderer
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('proctoring:notification', notification);
        }
      },
      onExit: (code, signal) => {
        // Notify renderer that monitoring stopped
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('proctoring:stopped', { code, signal });
        }
      }
    });

    return result;
  } catch (error) {
    console.error('Proctoring start error:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('proctoring:stop', async () => {
  try {
    const result = await processTerminator.gracefulTerminate(
      cameraMonitor.monitorProcessId,
      { timeout: 5000 }
    );
    return result;
  } catch (error) {
    console.error('Proctoring stop error:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('proctoring:status', () => {
  try {
    return cameraMonitor.getStatus();
  } catch (error) {
    console.error('Proctoring status error:', error);
    return {
      isMonitoring: false,
      error: error.message
    };
  }
});

ipcMain.handle('proctoring:getLogs', (event, type = 'stdout', lines = 50) => {
  try {
    const logs = cameraMonitor.getLogs(type, lines);
    return {
      success: true,
      logs
    };
  } catch (error) {
    console.error('Proctoring get logs error:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('proctoring:setParticipantImage', (event, imagePath) => {
  try {
    const success = cameraMonitor.setParticipantImage(imagePath);
    return {
      success,
      error: success ? null : 'Failed to set participant image'
    };
  } catch (error) {
    console.error('Proctoring set participant image error:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('proctoring:getAlertsJSON', async (event, sessionId) => {
  try {
    const logDir = cameraMonitor.getLogDirectory();
    if (!logDir) {
      return {
        success: false,
        error: 'No active session'
      };
    }

    const alertsData = await logWatcher.readAlertsJSON(
      path.join(logDir, `session_${sessionId}_alerts.json`)
    );

    return {
      success: !!alertsData,
      data: alertsData,
      error: alertsData ? null : 'Failed to read alerts file'
    };
  } catch (error) {
    console.error('Proctoring get alerts JSON error:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle app quitting
app.on('before-quit', async () => {
  // Stop all proctoring processes
  console.log('App quitting - cleaning up processes');
  try {
    await processTerminator.terminateAll({ graceful: true, timeout: 3000 });
  } catch (error) {
    console.error('Error during cleanup:', error);
  }
});

// Security: Prevent navigation to external URLs
app.on('web-contents-created', (event, contents) => {
  contents.on('will-navigate', (event, navigationUrl) => {
    const parsedUrl = new URL(navigationUrl);
    
    // Allow navigation only to localhost in dev mode
    if (isDev && parsedUrl.origin === 'http://localhost:5173') {
      return;
    }
    
    // Prevent all other navigation
    if (parsedUrl.origin !== 'file://') {
      event.preventDefault();
    }
  });
});
