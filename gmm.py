import numpy as np
import cv2
from scipy.stats import norm

# 2 is most probable and 0 is least probable

cap = cv2.VideoCapture(0)
_,frame = cap.read()

row,col,_ = frame.shape

mean = np.zeros([3,row,col],np.float64)

variance = np.zeros([3,row,col],np.float64)
variance[:,:,:] = 400

omega = np.zeros([3,row,col],np.float64)
omega[0,:,:],omega[1,:,:],omega[2,:,:] = 0.1,0.2,0.7

omega_by_sigma = np.zeros([3,row,col],np.float64)

foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)

gauss_fit_index = np.zeros([row,col])

alpha = 0.3
T = 0.8

while cap.isOpened():
    _,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray.astype(np.float64)

    variance[0] = np.where(variance[0] < 0, 400, variance[0])
    variance[1] = np.where(variance[1] < 0, 400, variance[1])
    variance[2] = np.where(variance[2] < 0, 400, variance[2])
    sigma1 = np.sqrt(variance[0])
    sigma2 = np.sqrt(variance[1])
    sigma3 = np.sqrt(variance[2])

    compare_val_1 = cv2.absdiff(frame_gray,mean[0])
    compare_val_2 = cv2.absdiff(frame_gray,mean[1])
    compare_val_3 = cv2.absdiff(frame_gray,mean[2])

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3

    gauss_fit_index1 = np.where(compare_val_1 <= value1)
    gauss_not_fit_index1 = np.where(compare_val_1 > value1)
    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)
    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)

    match_index = np.zeros([row,col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)

    rho = alpha * norm.pdf(frame_gray[gauss_fit_index1], mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
    constant = rho * ((frame_gray[gauss_fit_index1] - mean[0][gauss_fit_index1]) ** 2)
    constant = np.where(constant<0.00000001,0,constant)
    mean[0][gauss_fit_index1] = (1 - rho) * mean[0][gauss_fit_index1] + rho * frame_gray[gauss_fit_index1]
    variance[0][gauss_fit_index1] = (1 - rho) * variance[0][gauss_fit_index1] + constant
    omega[0][gauss_fit_index1] = (1 - alpha) * omega[0][gauss_fit_index1] + alpha
    omega[0][gauss_not_fit_index1] = (1 - alpha) * omega[0][gauss_not_fit_index1]

    rho = alpha * norm.pdf(frame_gray[gauss_fit_index2], mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
    constant = rho * ((frame_gray[gauss_fit_index2] - mean[1][gauss_fit_index2]) ** 2)
    constant = np.where(constant < 0.00000001, 0, constant)
    mean[1][gauss_fit_index2] = (1 - rho) * mean[1][gauss_fit_index2] + rho * frame_gray[gauss_fit_index2]
    variance[1][gauss_fit_index2] = (1 - rho) * variance[1][gauss_fit_index2] + rho * constant
    omega[1][gauss_fit_index2] = (1 - alpha) * omega[1][gauss_fit_index2] + alpha
    omega[1][gauss_not_fit_index2] = (1 - alpha) * omega[1][gauss_not_fit_index2]

    rho = alpha * norm.pdf(frame_gray[gauss_fit_index3], mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
    constant = rho * ((frame_gray[gauss_fit_index3] - mean[2][gauss_fit_index3]) ** 2)
    constant = np.where(constant < 0.00000001, 0, constant)
    mean[2][gauss_fit_index3] = (1 - rho) * mean[2][gauss_fit_index3] + rho * frame_gray[gauss_fit_index3]
    variance[2][gauss_fit_index3] = (1 - rho) * variance[2][gauss_fit_index3] + constant
    omega[2][gauss_fit_index3] = (1 - alpha) * omega[2][gauss_fit_index3] + alpha
    omega[2][gauss_not_fit_index3] = (1 - alpha) * omega[2][gauss_not_fit_index3]

    mean[0][not_match_index] = frame_gray[not_match_index]
    variance[0][not_match_index] = 200
    omega[0][not_match_index] = 0.01

    # normalise omega
    sum = np.sum(omega,axis=0)
    omega = omega/sum

    omega_by_sigma[0] = omega[0] / sigma1mean
    omega_by_sigma[1] = omega[1] / sigma2
    omega_by_sigma[2] = omega[2] / sigma3

    index = np.argsort(omega_by_sigma,axis=0)
    omega_by_sigma = np.take_along_axis(omega_by_sigma,index,axis=0)

    mean = np.take_along_axis(mean,index,axis=0)
    variance = np.take_along_axis(variance,index,axis=0)
    omega = np.take_along_axis(omega,index,axis=0)

    sigma1 = np.sqrt(variance[0])
    sigma2 = np.sqrt(variance[1])
    sigma3 = np.sqrt(variance[2])

    compare_val_1 = cv2.absdiff(frame_gray, mean[0])
    compare_val_2 = cv2.absdiff(frame_gray, mean[1])
    compare_val_3 = cv2.absdiff(frame_gray, mean[2])

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3
    frame_gray = frame_gray.astype(np.uint8)

    fore_index1 = np.where(omega[2]>T)
    fore_index2 = np.where(((omega[2]+omega[1])>T) & (omega[2]<T))
    temp = np.zeros([row,col])
    temp[fore_index1] = 1
    index = np.where(compare_val_3<=value3)
    temp[index] = temp[index]+1
    index2 = np.where(temp==2)
    background[index2] = frame_gray[index2]

    temp = np.zeros([row,col])
    temp[fore_index2] = 1
    index = np.where((compare_val_3<=value3)|(compare_val_2<=value2))
    temp[index] = temp[index]+1
    index2 = np.where(temp==2)
    background[index] = frame_gray[index]
    index = np.where(variance[2]<0)
    print(index)
    cv2.imshow('BACKGROUND',background)
    cv2.imshow('frame',frame_gray)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
