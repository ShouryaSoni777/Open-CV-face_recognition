import cv2
import os
import time
import shutil
import pathlib

cap = cv2.VideoCapture(0)

# check_dir = os.path.join("/A.I face recognition","Real Time Face Recognition")

check_dir = pathlib.Path(__file__).absolute()
check_dir = str(check_dir)
check_dir = check_dir.split("w")
check_dir = check_dir[0]

# path_join = os.path.join(check_dir)

dirs = os.listdir(check_dir)

if "webcam_labels" not in dirs:
    os.mkdir(f"{check_dir}\webcam_labels")

name = input("Write Your Name : ")

dir_to_check = f"{check_dir}\webcam_labels"

persons = os.listdir(dir_to_check)
if name not in persons:
    path = os.path.join(f"{check_dir}\webcam_labels",name)
    try:
        os.mkdir(path=path)
        print("Dir created succesfully")

    except Exception:
        print("Dir Already Exists")

for i in range(1,101):
    ret, photo = cap.read()

    cv2.imwrite(f"{name}{i}.jpg", photo)

    try:
        shutil.move(f"{name}{i}.jpg", f"{check_dir}\webcam_labels\%s"%name)

    except Exception:
        print("File already Exists")
        os.remove(f"{name}{i}.jpg")
        break

cap.release()

import webcam_trainer
