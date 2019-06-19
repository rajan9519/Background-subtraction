import cv2
import numpy as np

def nothing(x):
    pass
# creating trackbar for values to be subtrackted
cv2.namedWindow('diff')
cv2.createTrackbar('val','diff',0,255,nothing)

# creating video element
cap = cv2.VideoCapture(0)
_,frame = cap.read()

#getting the shape of the frame capture which will be used for creating an array of resultant image which will store the diff
frame = cv2.flip(frame,1)

image1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
row,col = image1.shape
data_type = image1.dtype
res = np.zeros([row,col,1],data_type)
while True:
    _,image2 = cap.read()
    image2 = cv2.flip(image2, 1)

    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    val = cv2.getTrackbarPos('val','diff')

    res = image1-image2
    cv2.imshow('image',res)
    res = np.where(res<val,255,0)

    cv2.imshow('diff',res)

    image1 = image2

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
