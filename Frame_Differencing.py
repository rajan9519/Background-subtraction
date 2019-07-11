import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('diff')

# creating trackbar for values to be subtracted
cv2.createTrackbar('min_val','diff',0,255,nothing)
cv2.createTrackbar('max_val','diff',0,255,nothing)

# creating video element
cap = cv2.VideoCapture(0)
_,frame = cap.read()

image1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#getting the shape of the frame capture which will be used for creating an array of resultant image which will store the diff
row,col = image1.shape
res = np.zeros([row,col,1],np.uint8)

# converting data type integers 255 and 0 to uint8 type
a = np.uint8([255])
b = np.uint8([0])
while True:
    _,image2 = cap.read()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # getting threshold values from trackbar according to the lightning condition
    min_val = cv2.getTrackbarPos('min_val','diff')
    max_val = cv2.getTrackbarPos('max_val', 'diff')

    # using cv2.absdiff instead of image1 - image2 because 1 - 2 will give 255 which is not expected
    res = cv2.absdiff(image1,image2)
    cv2.imshow('image',res)

    # creating mask
    res = np.where((min_val<res) & (max_val>res),a,b)
    res = cv2.bitwise_and(image2,image2,mask=res)
    cv2.imshow('diff',res)

    # assigning new new to the previous frame
    image1 = image2

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
