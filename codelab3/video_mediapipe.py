import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Process video file
cap = cv2.VideoCapture('video_rostros.mp4')
prev = time.time()
frames = 0

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        
        # Draw face detections
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        frames += 1
        if frames % 10 == 0:
            now = time.time()
            fps = 10 / (now - prev)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display or encode frame
        cv2.imshow("MediaPipe Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()

# Process webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        cv2.imshow("MediaPipe Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
