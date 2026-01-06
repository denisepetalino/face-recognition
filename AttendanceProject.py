import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
imageNames = []
myList = os.listdir(path)

# iterate through folder and obtain only the names of the images
for name in myList:
    currentImg = cv2.imread(f'{path}/{name}')
    images.append(currentImg)
    imageNames.append(os.path.splitext(name)[0])

# iterate through images, convert into rgb, find encodings and append to list
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateStr = now.strftime("%d/%m/%Y %H:%M:%S")
            f.writelines(f'\n{name},{dateStr}\n')

encodeListKnown = findEncodings(images)
print('Encoding complete!')

# find matches with our encodings using webcam
cap = cv2.VideoCapture(0)
# capture each frame
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img,(0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # find locations and encodings of faces in webcam
    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodesCurrentFrame = face_recognition.face_encodings(imgSmall, facesCurrentFrame)

    # iterate through all faces found in current frame and compare with encodeListKnown
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        # lowest value in faceDis = best match
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            matchName = imageNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            # needs to be resized to original
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            # display a box around identified matching face
            cv2.rectangle(img, (y1,x1), (y2,x2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img, matchName, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            markAttendance(matchName)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
