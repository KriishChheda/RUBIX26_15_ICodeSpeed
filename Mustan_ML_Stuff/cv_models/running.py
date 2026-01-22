from ultralytics import YOLO
import cv2

# Load the phone detection model
model = YOLO('Mustan_ML_Stuff/cv_models/phone.pt')

# Option 1: Run inference on an image
results = model('path/to/your/image.jpg')

# Display results
for result in results:
    # Get bounding boxes
    boxes = result.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        
        print(f"Detected phone at [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}] with confidence {confidence:.2f}")
    
    # Show annotated image
    annotated_frame = result.plot()
    cv2.imshow('Phone Detection', annotated_frame)
    cv2.waitKey(0)

# Option 2: Run inference on webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, conf=0.5)  # confidence threshold
    
    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow('Phone Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Option 3: Run on video file
results = model('path/to/video.mp4', save=True)  # saves output video