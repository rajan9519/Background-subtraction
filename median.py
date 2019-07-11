import cv2
import numpy as np

# creating video element
cap = cv2.VideoCapture('cars.mp4')

# images from which Background to be estimated
images = []

#taking 13 frames to estimate the background
for i in range(13):
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    images.append(frame)

# getting shape of the frame to create background
row,col = frame.shape
background = np.zeros([row,col],np.uint8)
background = np.median(images,axis=0)

# by median openration data type of background changes so again change it to uint8
background = background.astype(np.uint8)
res = np.zeros([row,col],np.uint8)

# converting interger 0 and 255 to type uint8
a = np.uint8([255])
b = np.uint8([0])

# initialising i so that we can replace frames from images to get new frames
i = 0

# creating different kernels for erode and dilate openration. bigger for erode and smaller for dilate
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,4))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,6))
while cap.isOpened():
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    images[i%13] = frame1
    background = np.median(images, axis=0)
    background = background.astype(np.uint8)

    # taking absolute difference otherwise having trouble in setting a particular value of threshold used in np.where
    res = cv2.absdiff(frame1,background)
    res = np.where(res>20,a,b)
    res = cv2.morphologyEx(res,cv2.MORPH_ERODE,kernel2)
    res = cv2.morphologyEx(res, cv2.MORPH_DILATE, kernel1)

    # to get the colored part of the generated mask res
    res = cv2.bitwise_and(frame,frame,mask=res)
    cv2.imshow('median',res)
    cv2.imshow('background',background)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    i = i+1
cap.release()
cv2.destroyAllWindows()
