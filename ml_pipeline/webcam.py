import cv2
import time
import numpy as np
from ml_pipeline.predict import predict_image, load_predictor

def run_webcam_recognition(camera_id=0):
    print("\n[ASL Vision] Initializing ASL Real-Time Webcam Recognition...")
    try:
        load_predictor()
        print("[ASL Vision] Model loaded successfully!")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Cannot open webcam device #{camera_id}. Please check your camera connection or permissions.")
        return

    print("==================================================")
    print("ASL Real-Time Recognition Active")
    print("Place your hand gesture inside the green bounding box.")
    print("Keys: [q] Quit  |  [c] Toggle Recognition  |  [s] Save Frame")
    print("==================================================")

    # Frame properties
    box_size = 250
    active_prediction = True
    prev_time = time.time()
    last_pred = {"predicted_class": "-", "confidence": 0.0}

    frame_count = 0
    predict_every_n_frames = 2  # run inference every 2 frames for smooth 30+ FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape

        # Calculate bounding box coordinates (centered right)
        x1 = w - box_size - 40
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]

        frame_count += 1
        if active_prediction and (frame_count % predict_every_n_frames == 0):
            try:
                last_pred = predict_image(roi)
            except Exception as e:
                pass

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        # Draw ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place Hand Here", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Header Panel
        cv2.rectangle(frame, (20, 20), (320, 150), (25, 25, 25), -1)
        cv2.rectangle(frame, (20, 20), (320, 150), (0, 255, 200), 2)

        pred_char = last_pred.get("predicted_class", "-")
        pred_conf = last_pred.get("confidence", 0.0) * 100

        cv2.putText(frame, "SignMaster ASL Vision", (30, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Prediction: {pred_char}", (30, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {pred_conf:.1f}%", (30, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("SignMaster - Real-Time ASL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            active_prediction = not active_prediction
            print(f"Recognition {'resumed' if active_prediction else 'paused'}")
        elif key == ord('s'):
            fname = f"asl_capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved snapshot to {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[ASL Vision] Closed camera stream.")

if __name__ == "__main__":
    run_webcam_recognition()
