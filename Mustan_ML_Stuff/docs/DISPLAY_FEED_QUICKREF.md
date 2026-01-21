# DISPLAY_FEED - Quick Reference

## What is DISPLAY_FEED?

A configuration variable that controls whether video frames are displayed during proctoring.

- **`True`** = Normal mode with video display (default)
- **`False`** = Background mode without video display

## Quick Usage

### Normal Mode (With Display)
```python
from modules import ProctorConfig, ProctorPipeline

ProctorConfig.DISPLAY_FEED = True  # Show frames (default)
proctor = ProctorPipeline(config=ProctorConfig)
proctor.run()  # Exit with 'q' or ESC
```

### Background Mode (No Display)
```python
from modules import ProctorConfig, ProctorPipeline

ProctorConfig.DISPLAY_FEED = False  # No frames shown
proctor = ProctorPipeline(config=ProctorConfig)
proctor.run()  # Exit with CTRL+C
```

## Run Scripts

### With Display
```bash
python proctor_main.py
```

### Without Display (Background)
```bash
python proctor_main_background.py
```

## Test Configuration
```bash
python test_display_feed.py
```

## What Changes in Background Mode?

| Feature | Normal | Background |
|---------|--------|------------|
| Camera feed captured | ✅ Yes | ✅ Yes |
| Detections run | ✅ Yes | ✅ Yes |
| Alerts logged | ✅ Yes | ✅ Yes |
| Display window | ✅ Yes | ❌ No |
| Frame rendering | ✅ Yes | ❌ No |
| Overlays/Annotations | ✅ Yes | ❌ No |
| Exit key (q/ESC) | ✅ Yes | ❌ No |
| Exit CTRL+C | ✅ Yes | ✅ Yes |

## Files Modified

1. `modules/config.py` - Added DISPLAY_FEED variable
2. `modules/camera_pipeline.py` - Conditional display init
3. `modules/proctor_pipeline.py` - Conditional rendering
4. `proctor_main_background.py` - Example script (NEW)
5. `test_display_feed.py` - Test suite (NEW)

## Documentation

- `README_BACKGROUND_MODE.md` - Full documentation
- `DISPLAY_FEED_IMPLEMENTATION.md` - Technical implementation details

## When to Use Background Mode

✅ Server deployments  
✅ Headless systems  
✅ Docker containers  
✅ API integrations  
✅ Resource optimization  
✅ Multiple instances  

## When to Use Normal Mode

✅ Development  
✅ Debugging  
✅ Live monitoring  
✅ User feedback  
✅ Visual verification  

---

**Default:** DISPLAY_FEED = True (normal mode with display)
