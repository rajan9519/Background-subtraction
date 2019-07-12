import numpy as np
import cv2


def nothing(x):
	pass


cap = cv2.VideoCapture(0)
images = []

cv2.namedWindow('tracker')
cv2.createTrackbar('val','tracker',50,255,nothing)
while True:
	
	ret,frame = cap.read()
	cv2.imshow('image',frame)
	dim = (500,500)
	frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA) 
#converting images into grayscale       
                                                
	

	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	 
        
       
	images.append(frame_gray)
	
	if len(images)==50:
	        images.pop(0)

	image = np.array(images)
	val = cv2.getTrackbarPos('val','tracker')
	image = np.mean(image,axis=0)
	image = image.astype(np.uint8)
	cv2.imshow('background',image)
	image = image.astype(np.uint8)
	foreground_image = cv2.absdiff(frame_gray,image)

	a = np.array([0],np.uint8)
	b = np.array([255],np.uint8)

	img = np.where(foreground_image>val,frame_gray,a)
	cv2.imshow('foreground',img)

	if cv2.waitKey(1) & 0xFF == 27:
		break


cap.release()

cv2.destroyAllWindows()		
