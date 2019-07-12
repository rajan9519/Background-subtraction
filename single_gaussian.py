import cv2
import numpy as np


cap = cv2.VideoCapture(0   )
ret,mean = cap.read()
mean =cv2.cvtColor(mean,cv2.COLOR_BGR2GRAY)
(col,row) = mean.shape[:2]

var = np.ones((col,row),np.uint8)
var[:col,:row] = 150
count =0
while True:

        ret,frame = cap.read()                                
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        
        alpha =0.25
        
        new_mean = (1-alpha)*mean + alpha*frame_gray       
        new_mean = new_mean.astype(np.uint8)
        new_var = (alpha)*(cv2.subtract(frame_gray,mean)**2) + (1-alpha)*(var)
       
        value  = cv2.absdiff(frame_gray,mean)
        value = value /np.sqrt(var)
       
        
        mean = np.where(value < 2.5,new_mean,mean)
        var = np.where(value < 2.5,new_var,var)
        a = np.uint8([255])
        b = np.uint8([0])
        background =np.where(value < 2.5,frame_gray,0)
        forground = np.where(value>=2.5,frame_gray,b)
        cv2.imshow('background',background)       
        kernel = np.ones((3,3),np.uint8)
        
        erode = cv2.erode(forground,kernel,iterations =2)
       # erode = cv2.absdiff(forground,background)
             
        cv2.imshow('forground',erode)
        
        if cv2.waitKey(5) & 0xFF == 27:
                break
                
cap.release()                
cv2.destroyAllWindows()
