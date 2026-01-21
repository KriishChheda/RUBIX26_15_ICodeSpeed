# Electron App Setup Guide

## Overview
The AI Proctor app is now configured as an Electron desktop application built with React + Vite.

## Installation

```bash
npm install
```

## Development

### Run the App
This starts both the Vite dev server and Electron app with hot reload:

```bash
npm run dev
```

The app will:
- Start Vite dev server on http://localhost:5173
- Wait for the server to be ready
- Launch Electron desktop app pointing to the dev server
- Enable hot module replacement for React components
- Open DevTools automatically

**Note**: The app only runs as an Electron desktop application. There is no separate web-only mode.

## Building

### Build for Current Platform
```bash
npm run build
```

### Build for Specific Platforms
```bash
npm run build:win   # Windows
npm run build:mac   # macOS
npm run build:linux # Linux
```

Built apps will be in the `release/` directory.

## Project Structure

```
Proctor_App/
├── electron/
│   ├── main.js          # Electron main process
│   └── preload.js       # Preload script (IPC bridge)
├── src/                 # React application source
├── public/              # Static assets
├── dist/                # Vite build output
├── release/             # Electron build output
├── electron-builder.json # Electron builder config
├── vite.config.js       # Vite configuration
└── package.json         # Dependencies and scripts
```

## Key Files

### electron/main.js
- Creates the main application window
- Handles app lifecycle events
- Manages security policies
- Routes between dev server (development) and built files (production)

### electron/preload.js
- Provides secure IPC communication between renderer and main process
- Exposes whitelisted channels to the React app via `window.electron`
- Maintains security through context isolation

### vite.config.js
- Configured for Electron with `base: './'` for relative paths
- Sets up aliases and build output

### electron-builder.json
- Configures app metadata and build settings
- Defines platform-specific build targets
- Specifies which files to include in builds

## Adding IPC Communication

### 1. Main Process (electron/main.js)
```javascript
ipcMain.handle('yourChannel', async (event, arg) => {
  // Handle request
  return result;
});
```

### 2. Preload Script (electron/preload.js)
Add channel to whitelist:
```javascript
const validChannels = ['yourChannel'];
```

### 3. Renderer Process (React)
```javascript
const result = await window.electron.invoke('yourChannel', data);
```

## Security Features

- **Context Isolation**: Enabled to prevent renderer from accessing Node.js APIs directly
- **Node Integration**: Disabled for security
- **Content Security Policy**: Configured in index.html
- **Navigation Protection**: External URLs are blocked
- **Preload Script**: Whitelisted IPC channels only

## Camera/Webcam Access

The app is configured to allow webcam access for proctoring features:
- Permissions are handled through the webPreferences
- Use `react-webcam` or native `getUserMedia` in your React components

## Troubleshooting

### Port Already in Use
If port 5173 is already in use, either:
- Stop the other process using that port
- Change the port in vite.config.js and update electron/main.js

### Electron Won't Start
- Ensure all dependencies are installed: `npm install`
- Check that Vite dev server is running before Electron starts
- Look for errors in the terminal

### Build Issues
- Make sure `dist/` folder exists after running `npm run build`
- Check that all assets are in the `public/` folder
- Verify electron-builder.json file paths are correct

## Next Steps

1. **Install dependencies**: `npm install`
2. **Run the app**: `npm run dev`
3. **Customize the app**: Update src/ with your React components
4. **Build for production**: `npm run build`
