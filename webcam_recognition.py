import cv2
import numpy as np
import os

people = []

haar = cv2.CascadeClassifier("F:\A.I face recognition\HAAR_CASCADES\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

features = np.load(r"Real Time Face Recognition\features_webcam.npy",allow_pickle=True)
labels = np.load(r"Real Time Face Recognition\labels_webcam.npy",allow_pickle=True)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"Real Time Face Recognition\face_trained_webcam.yml")

path_listed = os.listdir(path=r"Real Time Face Recognition\webcam_labels")

for item_person in path_listed:
    people.append(item_person)

while True:
    ret, frames = cap.read()

    gray_img = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    face_rects = haar.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=4)

    for (x, y, w, h) in face_rects:

        face_rectangle = gray_img[y:y+h, x:x+y]

        label, confidence = face_recognizer.predict(face_rectangle)

        print(f"{people[label]} with a confidence of {confidence}")

        cv2.putText(frames, str(people[label]), (x+w-120,y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,0),thickness=1)

        cv2.rectangle(frames, (x,y), (x+w,y+h), (255, 255, 0),thickness=2)

    cv2.imshow("detected faces", frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
