# DeepFace Backend Configuration Guide

## Overview

The face matching system now uses **DeepFace** with configurable backends, optimized to accept **pre-cropped faces** from MediaPipe detection. This eliminates redundant face detection, significantly improving performance.

## Key Optimization

**Pre-cropped Face Processing**: The system passes already-detected faces from MediaPipe directly to DeepFace's embedding extraction, bypassing DeepFace's internal face detection step using `enforce_detection=False` and `detector_backend='skip'`.

**Performance Gain**: ~2x faster face matching by eliminating redundant detection.

## Configuration Parameters

Located in `modules/config.py`:

```python
FACE_MATCHING_ENABLED = True
FACE_MATCHING_BACKEND = "Facenet512"  # Backend model
FACE_MATCHING_DISTANCE_METRIC = "cosine"  # Distance metric
FACE_MATCHING_THRESHOLD = 0.30  # Distance threshold (lower = stricter)
```

## Available Backends

### 1. **Facenet512** (Recommended) ⭐
- **Embedding Size**: 512 dimensions
- **Accuracy**: Very High
- **Speed**: Fast
- **Threshold (cosine)**: 0.30
- **Best For**: Production use, good balance of accuracy and speed

### 2. **VGG-Face**
- **Embedding Size**: 2622 dimensions
- **Accuracy**: High
- **Speed**: Moderate
- **Threshold (cosine)**: 0.40
- **Best For**: High accuracy requirements

### 3. **Facenet**
- **Embedding Size**: 128 dimensions
- **Accuracy**: High
- **Speed**: Very Fast
- **Threshold (cosine)**: 0.40
- **Best For**: Real-time applications requiring speed

### 4. **ArcFace**
- **Embedding Size**: 512 dimensions
- **Accuracy**: State-of-the-art
- **Speed**: Moderate
- **Threshold (cosine)**: 0.68
- **Best For**: Maximum accuracy, research applications

### 5. **OpenFace**
- **Embedding Size**: 128 dimensions
- **Accuracy**: Moderate
- **Speed**: Very Fast
- **Threshold (cosine)**: 0.10
- **Best For**: Lightweight applications

### 6. **Dlib**
- **Embedding Size**: 128 dimensions
- **Accuracy**: High
- **Speed**: Fast
- **Threshold (cosine)**: 0.07
- **Best For**: Traditional CV pipelines

### 7. **SFace**
- **Embedding Size**: 128 dimensions
- **Accuracy**: Moderate-High
- **Speed**: Fast
- **Threshold (cosine)**: 0.593
- **Best For**: Balanced performance

### 8. **DeepFace** (Facebook Model)
- **Embedding Size**: 4096 dimensions
- **Accuracy**: High
- **Speed**: Slow
- **Threshold (cosine)**: 0.23
- **Best For**: Research, not recommended for real-time

## Distance Metrics

### Cosine Distance (Recommended) ⭐
- **Range**: 0 to 2 (practical: 0 to 1)
- **Interpretation**: Lower = more similar
- **Formula**: `1 - cosine_similarity`
- **Best For**: Most face recognition tasks

### Euclidean Distance
- **Range**: 0 to ∞
- **Interpretation**: Lower = more similar
- **Formula**: `||embedding1 - embedding2||`
- **Best For**: Direct distance measurement

### Euclidean L2
- **Range**: 0 to 2
- **Interpretation**: Lower = more similar
- **Formula**: L2-normalized Euclidean distance
- **Best For**: Normalized comparisons

## Threshold Guidelines

### Understanding Thresholds
- **Lower threshold** = Stricter matching (fewer false positives, more false negatives)
- **Higher threshold** = Looser matching (more false positives, fewer false negatives)

### Recommended Thresholds (Cosine Distance)

| Backend | Conservative | Balanced | Lenient |
|---------|-------------|----------|---------|
| Facenet512 | 0.25 | 0.30 | 0.35 |
| VGG-Face | 0.35 | 0.40 | 0.45 |
| Facenet | 0.35 | 0.40 | 0.45 |
| ArcFace | 0.60 | 0.68 | 0.75 |
| Dlib | 0.05 | 0.07 | 0.10 |
| SFace | 0.50 | 0.593 | 0.65 |
| OpenFace | 0.08 | 0.10 | 0.12 |

### Proctoring-Specific Recommendations
- **High Security Exam**: Use conservative threshold (e.g., Facenet512 @ 0.25)
- **Standard Proctoring**: Use balanced threshold (e.g., Facenet512 @ 0.30)
- **Lenient Monitoring**: Use lenient threshold (e.g., Facenet512 @ 0.35)

## Usage Example

### Basic Configuration
```python
from modules import Config, FaceMatcher

# Configure backend
Config.FACE_MATCHING_BACKEND = "Facenet512"
Config.FACE_MATCHING_DISTANCE_METRIC = "cosine"
Config.FACE_MATCHING_THRESHOLD = 0.30

# Initialize matcher
face_matcher = FaceMatcher(
    model_name=Config.FACE_MATCHING_BACKEND,
    distance_metric=Config.FACE_MATCHING_DISTANCE_METRIC,
    distance_threshold=Config.FACE_MATCHING_THRESHOLD,
    participant_image_path="data/participant.png"
)

# Load model and participant embedding
face_matcher.load_model()
```

### Matching Pre-Cropped Faces
```python
# Get face from MediaPipe detector
face_meshes = face_detector.detect(frame)

# Extract face ROI (already cropped)
bbox = face_meshes[0]['bbox']
x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
face_roi = frame[y:y+h, x:x+w]

# Match without redundant detection
result = face_matcher.match_with_details(face_roi)

if result['matched']:
    print(f"✓ VERIFIED - Distance: {result['distance']:.4f}")
else:
    print(f"✗ UNVERIFIED - Distance: {result['distance']:.4f}")
```

## Performance Optimization Tips

### 1. Model Selection
- **For Speed**: Facenet, OpenFace, Dlib
- **For Accuracy**: ArcFace, VGG-Face, Facenet512
- **Balanced**: Facenet512 (recommended)

### 2. Frame Skipping
```python
Config.FRAME_SKIP = 2  # Process every 3rd frame
```

### 3. Threading
The pipeline uses threading for non-blocking face verification:
- Face detection runs on main thread
- Face matching runs on background thread with 0.5s timeout

### 4. Embedding Caching
Participant embedding is computed once at initialization and reused for all comparisons.

## Troubleshooting

### Import Errors
```bash
# Install DeepFace and dependencies
pip install deepface tensorflow

# For GPU support
pip install tensorflow-gpu
```

### Model Download Issues
- DeepFace downloads models on first use
- Models are cached in `~/.deepface/weights/`
- Ensure internet connection for first run

### Performance Issues
1. Switch to faster backend (Facenet, OpenFace)
2. Increase frame skip rate
3. Reduce camera resolution
4. Use GPU acceleration (if available)

### False Positives/Negatives
1. Adjust threshold (lower for stricter)
2. Try different backends
3. Ensure good lighting in participant image
4. Use high-quality reference image

## Technical Details

### How It Works
1. **MediaPipe** detects face and provides bounding box
2. Face ROI is cropped from frame
3. **DeepFace** extracts embedding from pre-cropped face (skips detection)
4. Embedding is compared with cached participant embedding
5. Distance is computed and compared against threshold

### Key Code Flow
```python
# In face_matcher.py
def _extract_embedding(self, face_roi):
    embedding_objs = DeepFace.represent(
        img_path=face_roi,
        model_name=self.model_name,
        enforce_detection=False,  # Skip detection
        detector_backend='skip',  # Skip backend entirely
        align=True,  # Still align for accuracy
        normalization='base'
    )
    return embedding_objs[0]['embedding']
```

## Dependencies

```txt
deepface >= 0.0.79
tensorflow >= 2.13.0
opencv-python >= 4.13.0
mediapipe >= 0.10.31
```

## References

- [DeepFace Documentation](https://github.com/serengil/deepface)
- [Face Recognition Models Comparison](https://github.com/serengil/deepface#face-recognition-models)
- [Distance Metrics Explained](https://github.com/serengil/deepface#distance-metrics)

---

**Last Updated**: January 2026
**Version**: 2.0 (DeepFace Integration)
