#                                                Background-subtraction
## MOTIVTION
To learn and implement different Backgroound-subtraction method's and understand the concept of Gaussian probability distribution function
## AIM
To implement different background-subtraction method's and also to implement the resarch paper based on  Adpative background-Mixture Modal for real time tracking 
## OVERVIEW
Background subtraction is a major preprocessing steps in many vision based applications. For example, consider the cases like visitor counter where a static camera takes the number of visitors entering or leaving the room, or a traffic camera extracting information about the vehicles etc.

![capture (1)](https://user-images.githubusercontent.com/50518930/61316081-0d907180-a7b5-11e9-9b0a-3e1f7508a348.png)

## Background Subtraction Methods:
During this project we perform different methods for subtracting background and foreground form the frame of video 
For this we have used python, opencv and numpy module( matrix operation) 

### 1) Frame Differencing:
This method is through the difference between two consecutive images to determine the presence of moving objects
### 2) Mean Filter:
In this method background is estimated by taking mean of the previous N frames. Once background is estimated, foreground is estimated by the difference of background and current frame.

![download (1)](https://user-images.githubusercontent.com/50518930/61317408-c3f55600-a7b7-11e9-9032-6e784cded33d.jpg)
![1111111111111](https://user-images.githubusercontent.com/50518930/61319680-8f37cd80-a7bc-11e9-922f-8649605683d4.jpg)


### 3) Median Filter:
It similar to mean filter method but instead of taking the mean we take the median of n frames

### 4) Median Approximation:
If a pixel in the current frame has a value larger than the corresponding background pixel, the background pixel is incremented by 1.
Likewise, if the current pixel is less than the background pixel, the background is decremented by one. The background eventually converges to an actual background.

### 5) Running Gaussian Average /single gaussian:
For every pixel, fit one Gaussian PDF distribution (µ,σ) on the most recent n frames (this gives the background PDF).To accommodate for change in background over time (e.g. due to illumination changes or non-static background objects), at every frame, every pixel's mean and variance must be updated.

### 6) Gaussian Mixture Model:
Model each pixel as mixture of gaussians and using an on-line approximation to update the model.The Gaussian distribution of the adaptive mixture model are then evaluate to determine which are most likely to result from background process.Each pixel is classifeid based on whether the gaussian distribution which represents it most effectively is considered part of background model.

![360px-normal-distribution-pdf-svg_orig](https://user-images.githubusercontent.com/50518930/61319551-4bdd5f00-a7bc-11e9-8082-75ee32ff90c8.png)



## APPLICATIONS:
Background modeling and background subtraction algorithms are very commonly used in vehicle detection systems.

### MENTORS
#### 1)Abhay Khandelwal
#### 2) Kalpit Jangid

### TEAM MEMBERS
#### 1) Saurabh Kemekar
#### 2) Rajan Kumar Singh
Website -: www.backsub.weebly.com
