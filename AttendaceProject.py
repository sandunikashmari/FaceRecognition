import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageAttendance'
Images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImage = cv2.imread(f'{path}/{cl}')       # Name of the Class(Image)
    Images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodinges(Images):
    encodeList = []
    for img in Images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendane (name):
    with open ('Attendance.csv', 'r+') as f:
        mydataList = f.readlines()
        nameList = []
        for line in mydataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H : %M : %S')
                f.writelines(f'\n{name}, {dtString}')


encodeListKnown = findEncodinges(Images)
#print(len(encodeListKnown))
print('Encoding Complete')

# 03) Find the matches of encoding

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0),2)
            cv2.rectangle(img,(x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendane(name)

        cv2.imshow('WebCam', img)
        cv2.waitKey(1)
