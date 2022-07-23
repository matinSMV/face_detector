import numpy as np
import cv2

face = cv2.CascadeClassifier (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye = cv2.CascadeClassifier (cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.1,5)
    #eyes = eye.detectMultiScale(gray,1.1,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w , y+h),(0,255,0),2)
        
    #for(x,y,w,h) in eyes:
        #cv2.rectangle(img,(x,y),(x+w , y+h),(0,255,0),2)
    
    cv2.imshow('img' , img)
    
    k = cv2.waitKey(30) & 0xff
    if k ==27:
        break
    
    
cap.release()
cv2.destroyAllWindows()
    