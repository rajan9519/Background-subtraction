import cv2
import numpy as np

# creating video element
cap = cv2.VideoCapture(0)
_,frame = cap.read()

# getting shape of the frame
row,col,channel = frame.shape

# initialising background and foreground
background = np.zeros([row,col],np.uint8)
foreground = np.zeros([row,col],np.uint8)

# converting data type of intergers 0 and 255 to uint8 type
a = np.uint8([255])
b = np.uint8([0])

# creating kernel for removing noise
kernel = np.ones([3,3],np.uint8)

while cap.isOpened() :
    _,frame1 = cap.read()
    frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    # applying algorithm of median approximation method to get estimated background
    background = np.where(frame>background,background+1,background-1)

    # using cv2.absdiff instead of background - frame, because 1 - 2 will give 255 which is not expected
    foreground = cv2.absdiff(background,frame)
    
    # setting a threshold value for removing noise and getting foreground
    foreground = np.where(foreground>40,a,b)
    
    # removing noise
    foreground = cv2.erode(foreground,kernel)
    foreground = cv2.dilate(foreground,kernel)
    # using bitwise and to get colored foreground
    foreground = cv2.bitwise_and(frame1,frame1,mask=foreground)
    cv2.imshow('background',background)
    cv2.imshow('foreground',foreground)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
