import cv2
import numpy as np
import keras

# Load emotion model
emotion_model = keras.models.load_model('emotion_detection_model.h5')

# Emotion labels (FER2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
vid_cam = cv2.VideoCapture(0)

if not vid_cam.isOpened():
    print("ERROR! NOT ABLE TO ACCESS CAMERA!")
    exit()

while True:
    capturing, frame = vid_cam.read()
    if not capturing:
        print("ERROR! CANNOT READ FRAME!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract face ROI
        face = gray[y:y+h, x:x+w]

        # Resize to 48x48
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        # Predict emotion
        prediction = emotion_model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        # Show emotion
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(frame, f"Faces detected: {len(faces)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow("Realtime Face & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_cam.release()
cv2.destroyAllWindows()