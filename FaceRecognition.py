#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np


# In[2]:


def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist=[]

    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]

        d=distance(test,ix)
        dist.append([d,iy])

    dk=sorted(dist,key=lambda x:x[0])[:k]

    labels=np.array(dk)[:,-1]

    output=np.unique(labels,return_counts=True)

    index=np.argmax(output[1])
    return output[0][index]


# In[3]:


capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip=0

face_data=[]
label=[]

class_id=0
names={}

path_of_dataset='./data_file/'


# In[4]:


for file in os.listdir(path_of_dataset):
    if file.endswith('.npy'):

        names[class_id]=file[:-4]

        print(f"File {file} has been loaded")
        data_item=np.load(path_of_dataset+file)
        face_data.append(data_item)


        target=class_id*np.ones((data_item.shape[0],))
        class_id +=1
        label.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_label=np.concatenate(label,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_label.shape)

trainset = np.concatenate((face_dataset,face_label),axis=1)
print(trainset.shape)


# In[5]:


while True:
    ret,frame=capture.read()
    

    if ret==False:
        cv2.waitKey(10)
        continue

    # gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
   

    for(x,y,w,h) in faces:

        offset=10
        # section_face = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        section_face = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        # print(section_face.shape)

        if section_face.shape[0] > 0 and section_face.shape[1] > 0:
            section_face = cv2.resize(section_face, (100, 100))
        else:
            print("Warning: Empty region detected.")
            continue
            
        section_face=cv2.resize(section_face,(100,100))

        output=knn(trainset,section_face.flatten())

        predicted_name=names[int(output)]


        
        cv2.putText(frame,predicted_name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(38, 119, 243),2,cv2.LINE_AA)
        # # cv2.rectangle(frame,(x,y),(x+w,y+h),(51,51,51),2)


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


       


    cv2.imshow("Faces: ",frame)
    
    stopKey=cv2.waitKey(1) & 0xFF
    if stopKey==ord('q') or stopKey==ord('Q') or cv2.getWindowProperty("Faces: ", cv2.WND_PROP_VISIBLE) < 1:
        break

    
       

    

capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




