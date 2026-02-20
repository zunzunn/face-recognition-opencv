import cv2
import face_recognition
import pickle

print("📸 Press SPACE to capture your face")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1)

    # SPACE to capture
    if key == 32:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)

        if len(faces) == 0:
            print("❌ No face detected. Try again.")
            continue

        encoding = face_recognition.face_encodings(rgb, faces)[0]

        with open("my_face.pkl", "wb") as f:
            pickle.dump(encoding, f)

        print("✅ Face registered successfully!")
        break

    # ESC to quit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()