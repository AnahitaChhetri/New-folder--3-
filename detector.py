import cv2
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
capture_vid=cv2.VideoCapture(0)
if not capture_vid.isOpened():
    print("ERROR! Cannot open webcam.")
while True:
    capt_frame, frame = capture_vid.read()
    if not capt_frame:
        print("Error! Cannot read frame!")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Realtime Face Detection", frame)
    if cv2.waitKey(1)==ord('q'):
        break

capture_vid.release()
cv2.destroyAllWindows()
