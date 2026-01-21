# Background Mode for Proctoring System

## Overview
The proctoring system now supports running in **background mode** where no video frames are displayed. This is useful for:
- Running proctoring as a background service
- Minimizing resource usage (no rendering overhead)
- Silent monitoring without visual feedback
- Server-side deployments

## Configuration

### Enable Background Mode
Set `DISPLAY_FEED = False` in your configuration:

```python
from modules import ProctorConfig

ProctorConfig.DISPLAY_FEED = False  # Disable frame display
```

### What Happens in Background Mode
When `DISPLAY_FEED` is `False`:
- ✅ Camera feed is captured normally
- ✅ All detection modules run (face, eye, verification)
- ✅ Alerts are logged to files
- ✅ Session data is saved
- ❌ No display window is created
- ❌ No frames are rendered
- ❌ No overlay text/annotations are drawn
- ❌ Exit via keyboard (use CTRL+C instead)

## Usage Examples

### Example 1: Background Mode Script
```python
from modules import ProctorPipeline, ProctorConfig

# Configure for background mode
ProctorConfig.DISPLAY_FEED = False
ProctorConfig.FACE_DETECT_ENABLE = True
ProctorConfig.FACE_MATCH_ENABLE = True
ProctorConfig.EYE_TRACKING_ENABLE = True

# Create and run pipeline
proctor = ProctorPipeline(config=ProctorConfig)
proctor.run()  # Runs until CTRL+C
```

### Example 2: Run Provided Background Script
```bash
python proctor_main_background.py
```

### Example 3: Normal Mode (With Display)
```python
from modules import ProctorPipeline, ProctorConfig

# Configure for normal mode (default)
ProctorConfig.DISPLAY_FEED = True  # Show frames

# Create and run pipeline
proctor = ProctorPipeline(config=ProctorConfig)
proctor.run()  # Runs until 'q' or ESC pressed
```

## Performance Benefits

| Feature | Display ON | Display OFF |
|---------|-----------|-------------|
| Frame Rendering | Yes | No |
| Overlay Drawing | Yes | No |
| Window Management | Yes | No |
| CPU Usage | Higher | Lower |
| Memory Usage | Higher | Lower |
| Detection Speed | Normal | Faster |

## Files Modified

1. **config.py**
   - Added `DISPLAY_FEED` configuration variable

2. **camera_pipeline.py**
   - Conditionally initializes display window
   - Skips frame display when `DISPLAY_FEED=False`
   - Handles exit differently in background mode

3. **proctor_pipeline.py**
   - Skips annotation drawing when `DISPLAY_FEED=False`
   - Skips overlay rendering when display is disabled
   - All detection logic continues to work

## Exit Methods

### Normal Mode (Display ON)
- Press `q` or `ESC` key

### Background Mode (Display OFF)
- Press `CTRL+C` (KeyboardInterrupt)
- Send termination signal to process

## Logging

All proctoring data is logged regardless of display mode:
- Session logs: `logs/proctoring/session_<id>.json`
- Alert logs: `logs/proctoring/session_<id>_alerts.json`
- Eye movements: `logs/eye_movements/eye_movements_<id>.jsonl`

## Integration Example

### Running as a Service
```python
import signal
import sys
from modules import ProctorPipeline, ProctorConfig

proctor = None

def signal_handler(sig, frame):
    print('\nShutting down proctoring...')
    if proctor:
        proctor.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Configure background mode
ProctorConfig.DISPLAY_FEED = False
proctor = ProctorPipeline(config=ProctorConfig)

print("Proctoring service started...")
proctor.run()
```

## Notes

- Background mode is ideal for production deployments
- All detection and logging features remain fully functional
- Display overhead is completely eliminated
- Suitable for headless servers and containerized environments
