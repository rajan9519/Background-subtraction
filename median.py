import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('median')

# creating video element
cap = cv2.VideoCapture('cars.mp4')

images = []
cv2.createTrackbar('val','median',0,255,nothing)
for i in range(13):
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    images.append(frame)
row,col = frame.shape
background = np.zeros([row,col],np.uint8)
background = np.median(images,axis=0)
background = background.astype(np.uint8)
res = np.zeros([row,col],np.uint8)
a = np.uint8([1])
b = np.uint8([0])
i = 0
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,4))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,10))
background = np.zeros([row, col], np.uint8)
while cap.isOpened():
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    images[i%13] = frame1
    background = np.median(images, axis=0)
    background = background.astype(np.uint8)

    val = cv2.getTrackbarPos('val','median')
    res = cv2.absdiff(frame1,background)
    res = np.where(res>val,a,b)
    res = cv2.medianBlur(res,5)
    res = cv2.morphologyEx(res,cv2.MORPH_ERODE,kernel2)
    res = cv2.morphologyEx(res, cv2.MORPH_DILATE, kernel1)

    res = cv2.bitwise_and(frame,frame,mask=res)
    cv2.imshow('median',res)
    cv2.imshow('background',background)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    i = i+1
cap.release()
cv2.destroyAllWindows()



cap.release()
cv2.destroyAllWindows()
