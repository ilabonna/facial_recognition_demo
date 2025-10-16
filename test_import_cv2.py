import cv2
import os
from mtcnn import MTCNN

# Define label and save path
name = "me"
save_dir = os.path.join("models", "data", name)
os.makedirs(save_dir, exist_ok=True)

# Initialize webcam and detector
cap = cv2.VideoCapture(0)
detector = MTCNN()

count = 0
max_images = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: flip horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Detect faces
    results = detector.detect_faces(frame)

    # Draw rectangles around faces
    for r in results:
        x, y, w, h = r['box']
        x, y = abs(x), abs(y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # Overlay counter text
    cv2.putText(frame, f"Images saved: {count}/{max_images}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Capture - press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and len(results) > 0:
        # Save only the first detected face
        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)
        face = frame[y:y+h, x:x+w]

        # Resize face to 160x160 (FaceNet default)
        face = cv2.resize(face, (160, 160))

        img_path = os.path.join(save_dir, f"{name}_{count:03d}.jpg")
        cv2.imwrite(img_path, face)
        count += 1
        print(f"Saved {img_path}")

        if count >= max_images:
            print("Reached max images. Exiting...")
            break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
