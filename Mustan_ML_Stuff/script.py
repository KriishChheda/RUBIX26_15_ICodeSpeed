import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"   # Path to your downloaded file
RISK_THRESHOLD = 50      # Alert if risk is above this
risk_score = 0           # Starting score

# --- 1. RISK LOGIC (Exact copy from training) ---
def calculate_risk(keypoints, current_score):
    """
    Calculates risk based on gaze direction.
    keypoints: [(inner_x, inner_y), (outer_x, outer_y), (pupil_x, pupil_y)]
    """
    if len(keypoints) < 3: return "LOST TRACK", current_score

    # Unpack points (Indices: 0=Inner, 1=Outer, 2=Pupil)
    inner, outer, pupil = keypoints[0], keypoints[1], keypoints[2]
    
    # Calculate geometric relationships
    eye_width = abs(outer[0] - inner[0]) + 1e-6 
    eye_center_y = (inner[1] + outer[1]) / 2
    
    # Ratios
    vertical_ratio = (pupil[1] - eye_center_y) / eye_width
    horizontal_ratio = (pupil[0] - inner[0]) / eye_width

    # Logic
    risk_inc = 0
    status = "Normal"
    color = (0, 255, 0) # Green

    # LOOKING DOWN (High Risk)
    if vertical_ratio > 0.15: 
        status = "LOOKING DOWN (RISK)"
        risk_inc = 2 + (vertical_ratio * 10) # Faster increase if looking deeper down
        color = (0, 0, 255) # Red

    # LOOKING UP (Thinking - Safe)
    elif vertical_ratio < -0.15:
        status = "THINKING (UP)"
        risk_inc = 0 # No penalty
        color = (255, 255, 0) # Cyan

    # SIDE GLANCE (Medium Risk)
    elif horizontal_ratio < 0.30 or horizontal_ratio > 0.70:
        status = "SIDE GLANCE (RISK)"
        risk_inc = 1
        color = (0, 165, 255) # Orange

    # CENTER (Safe)
    else:
        status = "FOCUSED"
        risk_inc = -0.5 # Cool down score slowly
        color = (0, 255, 0) # Green

    # Update Score (Clamp between 0 and 100)
    new_score = max(0, min(100, current_score + risk_inc))
    return status, new_score, color

# --- 2. MAIN APPLICATION ---
def main():
    global risk_score
    
    # Load Model
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    # Open Webcam (0 is usually default laptop cam)
    cap = cv2.VideoCapture(0)
    
    # Set Resolution (Optional, for speed)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Starting Eye Tracking... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run Inference
        # verbose=False keeps the terminal clean
        results = model.predict(frame, verbose=False,conf=0.5) 

        # Process Results
        for result in results:
            # Draw the standard skeleton (boxes and dots)
            frame = result.plot() 
            
            # Extract Keypoints for Risk Calculation
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                # Get the first detected eye
                kpts = result.keypoints.xy[0].cpu().numpy()
                
                # Calculate Risk
                status, risk_score, status_color = calculate_risk(kpts, risk_score)
                
                # --- DRAW UI ---
                # 1. Status Text (Top Left)
                cv2.putText(frame, f"STATUS: {status}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                # 2. Risk Score Bar (Bottom)
                # Background bar
                cv2.rectangle(frame, (50, 400), (550, 430), (50, 50, 50), -1)
                # Fill bar (Red if high risk, Green if low)
                bar_width = int((risk_score / 100) * 500)
                bar_color = (0, 0, 255) if risk_score > 50 else (0, 255, 0)
                cv2.rectangle(frame, (50, 400), (50 + bar_width, 430), bar_color, -1)
                
                # Score Text
                cv2.putText(frame, f"RISK SCORE: {int(risk_score)}%", (50, 390), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Alert if Risk is High
                if risk_score > 80:
                    cv2.putText(frame, "SUSPICIOUS ACTIVITY!", (150, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show the frame
        cv2.imshow('Eye Movement Risk Detector', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()