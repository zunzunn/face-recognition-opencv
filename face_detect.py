import cv2
import face_recognition
import pickle

# Load your saved face
with open("my_face.pkl", "rb") as f:
    known_encoding = pickle.load(f)

print(" Starting face recognition... Press ESC to exit")

cap = cv2.VideoCapture(0)

# ⚡ Lower camera resolution (big FPS boost)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

process_this_frame = True  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    process_this_frame = not process_this_frame

    if process_this_frame:
        face_locations = face_recognition.face_locations(
            rgb_small, model="hog"  
        )
        face_encodings = face_recognition.face_encodings(
            rgb_small, face_locations
        )

        results = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                [known_encoding], face_encoding
            )
            distance = face_recognition.face_distance(
                [known_encoding], face_encoding
            )[0]

            if matches[0]:
                results.append(("Face Detected", (0, 255, 0), distance))
            else:
                results.append(("Face Not Detected", (0, 0, 255), distance))

    
    if process_this_frame:
        draw_locations = face_locations
        draw_results = results

    for (top, right, bottom, left), (label, color, distance) in zip(
        draw_locations, draw_results
    ):
        top = int(top * 4)
        right = int(right * 4)
        bottom = int(bottom * 4)
        left = int(left * 4)

        display_text = (
            f"{label} ({1-distance:.2f})"
            if label == "Face Detected"
            else label
        )

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, top - 30), (right, top), color, -1)

        cv2.putText(
            frame,
            display_text,
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