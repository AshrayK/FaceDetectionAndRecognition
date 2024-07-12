#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np


# In[2]:


capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip=0
face_data=[]

path_of_dataset='./data_file/'
os.makedirs(path_of_dataset, exist_ok=True)


name_of_file=input("Enter your name: ")


while True:
    ret,frame=capture.read()  

    if ret==False:
        cv2.waitKey(10)
        continue
        
    #gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)

    for(x,y,w,h) in faces:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(51,51,51),2)

        x1,y1 = x,y
        x2,y2 = x+w,y+h
        r=int((5/100)*w)
        d=2*r
        color=(51,51,51)
        thickness=2
        # Top Left
        cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top Right
        cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom Left
        cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom Right
        cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        

        offset=10
        # section_face = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        section_face = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        section_face=cv2.resize(section_face,(100,100))
        if section_face.shape[0] > 0 and section_face.shape[1] > 0:
            section_face = cv2.resize(section_face, (100, 100))
        else:
            print("Warning: Empty region detected.")
            continue

        skip=skip+1
        if skip%5==0:
            face_data.append(section_face)
            print(len(face_data))

        cv2.imshow("Section of Capture Face",section_face)
            
    cv2.imshow("Capture Frame",frame)
    
    
    stopKey=cv2.waitKey(1) & 0xFF

    if stopKey==ord('q') or stopKey==ord('Q') or cv2.getWindowProperty("Capture Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

   


face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(path_of_dataset + name_of_file+".npy",face_data)
print("Image saved at "+path_of_dataset + name_of_file+".npy")

capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




