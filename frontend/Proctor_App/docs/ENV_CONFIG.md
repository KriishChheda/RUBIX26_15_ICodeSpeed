# Environment Configuration Guide

## Setup

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Adjust configuration** in `.env` file based on your setup

---

## Configuration Variables

### Development vs Production
- `NODE_ENV` - Set to `"development"` or `"production"`

### Frontend Server
- `VITE_DEV_SERVER_PORT` - Port for Vite dev server (default: `"5173"`)
- `VITE_API_URL` - Backend API URL (default: `"http://127.0.0.1:8000/api/"`)

### Python Monitor
- `PYTHON_PROJECT_PATH` - Path to ML project folder (default: `"Mustan_ML_stuff"`)
- `PYTHON_SCRIPT_NAME` - Python script filename (default: `"proctor_main_background.py"`)
- `PYTHON_EXECUTABLE_PATH` - Python executable (leave empty for auto-detect)
- `PARTICIPANT_IMAGE_PATH` - Participant image relative path (default: `"data/participant.png"`)

### Monitoring
- `ENABLE_MONITOR` - Enable/disable proctoring (default: `"true"`)
- `ALERT_STATE_FILENAME` - Alert file name (default: `"alert_state.txt"`)
- `LOG_DIRECTORY_PATH` - Log directory (default: `"logs/proctoring"`)
- `ALERT_POLL_INTERVAL` - Poll frequency in ms (default: `"200"`)

### Feature Flags
- `ENABLE_FACE_DETECTION` - Enable face detection (default: `"true"`)
- `ENABLE_FACE_MATCHING` - Enable face verification (default: `"true"`)
- `ENABLE_EYE_TRACKING` - Enable eye tracking (default: `"true"`)
- `ENABLE_PHONE_DETECTION` - Enable phone detection (default: `"false"`)

### Window Configuration
- `WINDOW_WIDTH` - Window width in pixels (default: `"1400"`)
- `WINDOW_HEIGHT` - Window height in pixels (default: `"900"`)
- `WINDOW_MIN_WIDTH` - Minimum width (default: `"1200"`)
- `WINDOW_MIN_HEIGHT` - Minimum height (default: `"700"`)
- `ENABLE_DEVTOOLS` - Show DevTools in dev mode (default: `"true"`)
- `ENABLE_KIOSK_MODE` - Enable kiosk/fullscreen mode (default: `"false"`)

### Security
- `ENABLE_WEB_SECURITY` - Enable web security (default: `"true"`)
- `ENABLE_CONTEXT_ISOLATION` - Enable context isolation (default: `"true"`)
- `ENABLE_NODE_INTEGRATION` - Enable Node in renderer (default: `"false"`)

---

## Important Notes

1. **Never commit `.env` file** - It's gitignored for security
2. **Always use double quotes** for string values
3. **Boolean values** should be `"true"` or `"false"` as strings
4. **Numeric values** should also be quoted as strings
5. **Empty values** use empty quotes: `""`

---

## Example Configurations

### Development Setup
```env
NODE_ENV="development"
VITE_DEV_SERVER_PORT="5173"
ENABLE_DEVTOOLS="true"
ENABLE_KIOSK_MODE="false"
ENABLE_MONITOR="true"
```

### Production Build
```env
NODE_ENV="production"
ENABLE_DEVTOOLS="false"
ENABLE_KIOSK_MODE="true"
ENABLE_MONITOR="true"
ENABLE_WEB_SECURITY="true"
```

### Testing Without Proctoring
```env
NODE_ENV="development"
ENABLE_MONITOR="false"
ENABLE_DEVTOOLS="true"
```

### Custom Python Path (Windows with conda)
```env
PYTHON_EXECUTABLE_PATH="C:/Users/YourUser/miniconda3/envs/proctor/python.exe"
PYTHON_PROJECT_PATH="Mustan_ML_stuff"
```

---

## Troubleshooting

### Python Not Found
- Leave `PYTHON_EXECUTABLE_PATH` empty for auto-detection
- Or provide full path to Python executable
- Check that virtual environment exists in project folder

### Monitor Not Starting
- Verify `ENABLE_MONITOR="true"`
- Check `PYTHON_PROJECT_PATH` points to correct folder
- Ensure `PYTHON_SCRIPT_NAME` exists in project folder

### DevTools Not Opening
- Set `ENABLE_DEVTOOLS="true"`
- Ensure `NODE_ENV="development"`
- DevTools won't open in production builds

### Alert Notifications Not Showing
- Check `ALERT_STATE_FILENAME` matches Python output file
- Verify `ALERT_POLL_INTERVAL` is reasonable (100-500ms)
- Ensure Python process is writing to correct location

---

## Related Files

- `.env.example` - Template with all available options
- `electron/main.js` - Loads and uses environment variables
- `electron/utils/camMonitorSpawn.js` - Python process spawner
- `src/services/api.js` - Uses `VITE_API_URL` for backend calls

---

## Need Help?

Check the documentation files:
- `ARCHITECTURE_FLOW.md` - System architecture overview
- `ALERT_STATE_SYSTEM.md` - Alert communication details
- `QUICKREF.md` - Quick reference guide
