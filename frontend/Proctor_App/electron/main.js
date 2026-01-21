const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const cameraMonitor = require('./utils/camMonitorSpawn');
const processTerminator = require('./utils/processTerminator');
const logWatcher = require('./utils/logWatcher');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '../.env') });

// Determine if running in development or production
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

// Configuration from environment variables
const config = {
  devServerPort: process.env.VITE_DEV_SERVER_PORT || '5173',
  windowWidth: parseInt(process.env.WINDOW_WIDTH || '1400'),
  windowHeight: parseInt(process.env.WINDOW_HEIGHT || '900'),
  windowMinWidth: parseInt(process.env.WINDOW_MIN_WIDTH || '1200'),
  windowMinHeight: parseInt(process.env.WINDOW_MIN_HEIGHT || '700'),
  enableDevTools: process.env.ENABLE_DEVTOOLS === 'true',
  enableKioskMode: process.env.ENABLE_KIOSK_MODE === 'true',
  enableWebSecurity: process.env.ENABLE_WEB_SECURITY !== 'false',
  enableContextIsolation: process.env.ENABLE_CONTEXT_ISOLATION !== 'false',
  enableNodeIntegration: process.env.ENABLE_NODE_INTEGRATION === 'true',
  enableMonitor: process.env.ENABLE_MONITOR !== 'false'
};

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: config.windowWidth,
    height: config.windowHeight,
    minWidth: config.windowMinWidth,
    minHeight: config.windowMinHeight,
    fullscreen: config.enableKioskMode,
    kiosk: config.enableKioskMode,
    webPreferences: {
      nodeIntegration: config.enableNodeIntegration,
      contextIsolation: config.enableContextIsolation,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: config.enableWebSecurity,
      enableWebSQL: false,
    },
    titleBarStyle: 'default',
    backgroundColor: '#0f172a',
    show: false,
    icon: path.join(__dirname, '../public/vite.svg')
  });

  // Show window when ready to avoid visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Load the app
  if (isDev) {
    // In development, load from Vite dev server
    mainWindow.loadURL(`http://localhost:${config.devServerPort}`);
    // Open DevTools in development if enabled
    if (config.enableDevTools) {
      mainWindow.webContents.openDevTools();
    }
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
    // Check if monitoring is enabled
    if (!config.enableMonitor) {
      return {
        success: false,
        error: 'Monitoring is disabled in configuration'
      };
    }

    // Initialize camera monitor with environment config
    const monitorConfig = {
      pythonPath: process.env.PYTHON_EXECUTABLE_PATH || undefined,
      scriptPath: process.env.PYTHON_SCRIPT_NAME 
        ? path.join(process.env.PYTHON_PROJECT_PATH || 'Mustan_ML_stuff', process.env.PYTHON_SCRIPT_NAME)
        : undefined,
      participantImagePath: process.env.PARTICIPANT_IMAGE_PATH
        ? path.join(process.env.PYTHON_PROJECT_PATH || 'Mustan_ML_stuff', process.env.PARTICIPANT_IMAGE_PATH)
        : undefined
    };

    const validation = cameraMonitor.initialize(monitorConfig);
    
    if (!validation.success) {
      return {
        success: false,
        error: 'Failed to initialize camera monitor',
        details: validation.checks
      };
    }

    // Start monitoring with feature flags from environment
    const result = cameraMonitor.startMonitoring({
      sessionId: options.sessionId,
      faceDetect: process.env.ENABLE_FACE_DETECTION !== 'false' && options.faceDetect,
      faceMatch: process.env.ENABLE_FACE_MATCHING !== 'false' && options.faceMatch,
      eyeTracking: process.env.ENABLE_EYE_TRACKING !== 'false' && options.eyeTracking,
      phoneDetect: process.env.ENABLE_PHONE_DETECTION === 'true' && options.phoneDetect,
      watchAlerts: true,
      watchLogs: false,
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
`http://localhost:${config.devServerPort}`
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
