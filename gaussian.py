import cv2
import numpy as np
from scipy.stats import norm
import math

# reading video from the webcam
cap = cv2.VideoCapture(0)
_, frame = cap.read()
# getting shape for other matrices
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
row,col = frame.shape
# defing other matrices to store variance,mean background etc.
variance = np.zeros([row,col], np.float64)
mean = np.zeros([row,col], np.float64)
background = np.zeros([row,col], np.uint8)
# converting integers to type uint8
a = np.uint8([0])
b = np.uint8([255])
# setting variance to high value
variance[:,:] = 200
#initialising alpha the learning rate
alpha = 0.9

while cap.isOpened():

    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # converting data type so that no error on line 37 for different type operation
    frame_gray = frame_gray.astype(np.float64)

    # standard deviation sigma
    sigma = np.sqrt(variance)

    background = np.where((frame_gray >= mean - 2.5*sigma) & (frame_gray < mean + 2.5*sigma),frame_gray,background)
    foreground = np.where((frame_gray >= mean - 2.5*sigma) & (frame_gray < mean + 2.5*sigma),a,b)

    new_mean = (1-alpha) * mean + alpha * frame_gray
    new_var = (1-alpha) * variance + alpha * (cv2.absdiff(frame_gray, mean) * cv2.absdiff(frame_gray, mean))

    mean = np.where((frame_gray >= mean - 2.5 * sigma) & (frame_gray < mean + 2.5 * sigma), new_mean, mean)
    variance = np.where((frame_gray >= mean - 2.5 * sigma) & (frame_gray < mean + 2.5 * sigma), new_var, variance)

    frame_gray = frame_gray.astype(np.uint8)
    cv2.imshow('foreground',background)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
