import cv2
import keras

emotion_model = keras.models.load_model('emotion_detection_model.h5')
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

vid_cam=cv2.VideoCapture(0)
if not vid_cam:
    print("ERROR! NOT ABLE TO ACCESS CAMERA!")
while True:
    capturing, frame= vid_cam.read()
    if not capturing:
        print("ERROR! CANNOT READ FRAME!")
    grey_colour=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_recognize=face_cascade.detectMultiScale(grey_colour, scaleFactor=1.1, minNeighbors=7)
    for (a, b, c, d) in face_recognize:
        cv2.rectangle(frame, (a,b), (a+c,b+d), (0,255,0), 2)
        prediction=emotion_model.predict(grey_colour)
        cv2.putText(frame, f"{prediction}", (100,30))
    cv2.putText(frame, f"Faces detected: {len(face_recognize)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Realtime Face Detection", frame)
    if cv2.waitKey(1)==ord('q'):
        break

vid_cam.release()
cv2.destroyAllWindows()
