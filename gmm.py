import numpy as np
import cv2

def normal_pdf(x, mu, sigma):
    y = 0
    try:
        y = 1.0 / ((sigma * (2.0 * np.pi) ** (1 / 2)) * np.exp((x - mu) ** 2 / (2.0 * (sigma ** 2))))
    except:
        RuntimeWarning
    print(y)
    return y

def sort(x):
    return 0

# 1 is most probable and 3 is least probable

cap = cv2.VideoCapture(0)
_,frame = cap.read()

row,col,_ = frame.shape

mean_g1 = np.zeros([row,col],np.float64)
mean_g2 = np.zeros([row,col],np.float64)
mean_g3 = np.zeros([row,col],np.float64)

variance_g1 = np.ones([row,col],np.float64)
variance_g2 = np.ones([row,col],np.float64)
variance_g3 = np.ones([row,col],np.float64)
variance_g1[:,:],variance_g2[:,:],variance_g3[:,:] = 200,200,200

omega_g1 = np.ones([row,col],np.float64)
omega_g2 = np.ones([row,col],np.float64)
omega_g3 = np.ones([row,col],np.float64)
omega_g1[:,:],omega_g2[:,:],omega_g3[:,:] = 1,0,0

foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)

gauss_fit_index = np.zeros([row,col])

alpha = 0.8
T = 0.7

while cap.isOpened():
    _,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray.astype(np.float64)

    sigma1 = np.sqrt(variance_g1)
    sigma2 = np.sqrt(variance_g2)
    sigma3 = np.sqrt(variance_g3)

    compare_val_1 = cv2.absdiff(frame_gray,mean_g1)
    compare_val_2 = cv2.absdiff(frame_gray,mean_g2)
    compare_val_3 = cv2.absdiff(frame_gray,mean_g3)

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

    rho = alpha * normal_pdf(frame_gray, mean_g1, sigma1)
    mean_g1[gauss_fit_index1] = (1 - rho[gauss_fit_index1]) * mean_g1[gauss_fit_index1] + rho[gauss_fit_index1] * frame_gray[gauss_fit_index1]
    variance_g1[gauss_fit_index1] = (1 - rho[gauss_fit_index1]) * (sigma1[gauss_fit_index1] ** 2) + rho[gauss_fit_index1] * ((frame_gray[gauss_fit_index1] - mean_g1[gauss_fit_index1]) ** 2)
    omega_g1[gauss_fit_index1] = (1 - alpha) * omega_g1[gauss_fit_index1] + alpha
    omega_g1[gauss_not_fit_index1] = (1 - alpha) * omega_g1[gauss_not_fit_index1]

    rho = alpha * normal_pdf(frame_gray, mean_g2, sigma2)
    mean_g2[gauss_fit_index2] = (1 - rho[gauss_fit_index2]) * mean_g2[gauss_fit_index2] + rho[gauss_fit_index2] * frame_gray[gauss_fit_index2]
    variance_g2[gauss_fit_index2] = (1 - rho[gauss_fit_index2]) * (sigma2[gauss_fit_index2] ** 2) + rho[gauss_fit_index2] * ((frame_gray[gauss_fit_index2] - mean_g2[gauss_fit_index2]) ** 2)
    omega_g2[gauss_fit_index2] = (1 - alpha) * omega_g2[gauss_fit_index2] + alpha
    omega_g2[gauss_not_fit_index2] = (1 - alpha) * omega_g2[gauss_not_fit_index2]

    rho = alpha * normal_pdf(frame_gray, mean_g3, sigma3)
    mean_g3[gauss_fit_index3] = (1 - rho[gauss_fit_index3]) * mean_g3[gauss_fit_index3] + rho[gauss_fit_index3] * frame_gray[gauss_fit_index3]
    variance_g3[gauss_fit_index3] = (1 - rho[gauss_fit_index3]) * (sigma3[gauss_fit_index3] ** 2) + rho[gauss_fit_index3] * ((frame_gray[gauss_fit_index3] - mean_g3[gauss_fit_index3]) ** 2)
    omega_g3[gauss_fit_index3] = (1 - alpha) * omega_g1[gauss_fit_index3] + alpha
    omega_g3[gauss_not_fit_index3] = (1 - alpha) * omega_g1[gauss_not_fit_index3]

    mean_g3[not_match_index] = frame_gray[not_match_index]
    variance_g3[not_match_index] = 200
    omega_g3[not_match_index] = 0.01

    # normalise omega
    sum = omega_g1 + omega_g2 + omega_g3
    omega_g1 = omega_g1/sum
    omega_g2 = omega_g2/sum
    omega_g3 = omega_g3/sum

    omega_by_sigma_1 = omega_g1/sigma1
    omega_by_sigma_2 = omega_g2 / sigma2
    omega_by_sigma_3 = omega_g3 / sigma3

    #m_1_2_3 = np.where((omega_by_sigma_1 >= omega_by_sigma_2) & (omega_by_sigma_2 >= omega_by_sigma_3))
    m_2_1_3 = np.where((omega_by_sigma_2 >= omega_by_sigma_1) & (omega_by_sigma_1 >= omega_by_sigma_3))
    m_3_2_1 = np.where((omega_by_sigma_3 >= omega_by_sigma_2) & (omega_by_sigma_2 >= omega_by_sigma_1))
    m_1_3_2 = np.where((omega_by_sigma_1 >= omega_by_sigma_3) & (omega_by_sigma_3 >= omega_by_sigma_2))
    m_2_3_1 = np.where((omega_by_sigma_2 >= omega_by_sigma_3) & (omega_by_sigma_3 >= omega_by_sigma_1))
    m_3_1_2 = np.where((omega_by_sigma_3 >= omega_by_sigma_1) & (omega_by_sigma_1 >= omega_by_sigma_2))

    temp = mean_g1[m_2_1_3]
    mean_g1[m_2_1_3] = mean_g2[m_2_1_3]
    mean_g2[m_2_1_3] = temp

    temp = mean_g3[m_3_2_1]
    mean_g3[m_3_2_1] = mean_g1[m_3_2_1]
    mean_g1[m_3_2_1] = temp

    temp = mean_g3[m_1_3_2]
    mean_g3[m_1_3_2] = mean_g2[m_1_3_2]
    mean_g2[m_1_3_2] = temp

    temp = mean_g1[m_2_3_1]
    mean_g1[m_2_3_1] = mean_g2[m_2_3_1]
    mean_g2[m_2_3_1] = mean_g3[m_2_3_1]
    mean_g3[m_2_3_1] = temp

    temp = mean_g3[m_3_1_2]
    mean_g3[m_3_1_2] = mean_g2[m_3_1_2]
    mean_g2[m_3_1_2] = mean_g1[m_3_1_2]
    mean_g1[m_3_1_2] = temp

    temp = variance_g1[m_2_1_3]
    variance_g1[m_2_1_3] = variance_g2[m_2_1_3]
    variance_g2[m_2_1_3] = temp

    temp = variance_g3[m_3_2_1]
    variance_g3[m_3_2_1] = variance_g1[m_3_2_1]
    variance_g1[m_3_2_1] = temp

    temp = variance_g3[m_1_3_2]
    variance_g3[m_1_3_2] = variance_g2[m_1_3_2]
    variance_g2[m_1_3_2] = temp

    temp = variance_g1[m_2_3_1]
    variance_g1[m_2_3_1] = variance_g2[m_2_3_1]
    variance_g2[m_2_3_1] = variance_g3[m_2_3_1]
    variance_g3[m_2_3_1] = temp

    temp = variance_g3[m_3_1_2]
    variance_g3[m_3_1_2] = variance_g2[m_3_1_2]
    variance_g2[m_3_1_2] = variance_g1[m_3_1_2]
    variance_g1[m_3_1_2] = temp

    temp = omega_g1[m_2_1_3]
    omega_g1[m_2_1_3] = omega_g2[m_2_1_3]
    omega_g2[m_2_1_3] = temp

    temp = omega_g3[m_3_2_1]
    omega_g3[m_3_2_1] = omega_g1[m_3_2_1]
    omega_g1[m_3_2_1] = temp

    temp = omega_g3[m_1_3_2]
    omega_g3[m_1_3_2] = omega_g2[m_1_3_2]
    omega_g2[m_1_3_2] = temp

    temp = omega_g1[m_2_3_1]
    omega_g1[m_2_3_1] = omega_g2[m_2_3_1]
    omega_g2[m_2_3_1] = omega_g3[m_2_3_1]
    omega_g3[m_2_3_1] = temp

    temp = omega_g3[m_3_1_2]
    omega_g3[m_3_1_2] = omega_g2[m_3_1_2]
    omega_g2[m_3_1_2] = omega_g1[m_3_1_2]
    omega_g1[m_3_1_2] = temp

    sigma1 = np.sqrt(variance_g1)
    sigma2 = np.sqrt(variance_g2)
    #sigma3 = np.sqrt(variance_g3)

    compare_val_1 = cv2.absdiff(frame_gray, mean_g1)
    compare_val_2 = cv2.absdiff(frame_gray, mean_g2)
    #compare_val_3 = cv2.absdiff(frame_gray, mean_g3)

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    #value3 = 2.5 * sigma3

    fore_index = np.where((compare_val_1>value1))

    frame_gray = frame_gray.astype(np.uint8)
    frame_gray[fore_index] = np.uint([0])

    cv2.imshow('frame',frame_gray)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
