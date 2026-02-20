import cv2
import face_recognition
import pickle

# Load your saved face
with open("my_face.pkl", "rb") as f:
    known_encoding = pickle.load(f)

print("🚀 Starting face recognition... Press ESC to exit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed (important!)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back up
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

        if matches[0]:
            color = (0, 255, 0)
            label = f"Face Detected ({1-distance:.2f})"
        else:
            color = (0, 0, 255)
            label = "Face Not Detected"

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw label background
        cv2.rectangle(frame, (left, top - 30), (right, top), color, -1)

        # Put text
        cv2.putText(
            frame,
            label,
            (left + 5, top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()