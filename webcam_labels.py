import cv2
import os
import time
import shutil
import pathlib

cap = cv2.VideoCapture(0)

# check_dir = os.path.join("/A.I face recognition","Real Time Face Recognition")

check_dir = pathlib.Path().absolute()

path_join = os.path.join(check_dir,"Real Time Face Recognition")

dirs = os.listdir(path_join)

if "webcam_labels" not in dirs:
    os.mkdir("Real Time Face Recognition\webcam_labels")

name = input("Write Your Name : ")

dir_to_check = r"Real Time Face Recognition\webcam_labels"

persons = os.listdir(dir_to_check)
if name not in persons:
    path = os.path.join("Real Time Face Recognition\webcam_labels",name)
    try:
        os.mkdir(path=path)
        print("Dir created succesfully")

    except Exception:
        print("Dir Already Exists")

for i in range(1,11):
    ret, photo = cap.read()

    cv2.imwrite(f"{name}{i}.jpg", photo)

    try:
        shutil.move(f"{name}{i}.jpg", "Real Time Face Recognition\webcam_labels\%s"%name)

    except Exception:
        print("File already Exists")
        os.remove(f"{name}{i}.jpg")

cap.release()
