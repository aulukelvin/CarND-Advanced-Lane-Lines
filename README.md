# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistortedChessBoard.png "undistortedChessBoard.png"
[image2]: ./output_images/undistortedRoad.png "undistortedRoad.png"
[image3]: ./output_images/HLS.png "HLS.png"
[image4]: ./output_images/combinedbinary.png "combinedbinary.png"
[image5]: ./output_images/histogram.png "histogram.png"
[image6]: ./output_images/birdsEyeView.png "birdsEyeView.png"
[image7]: ./output_images/slidingWindow.png "slidingWindow.png"
[image8]: ./output_images/laneFinder.png "laneFinder.png"
[image9]: ./output_images/pavedBack.png "pavedBack.png"

### [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Code structure

|  Path  | Description  |
|---|---|
| /README.md  | writeup  |
| /CarND_Advanced_lane_finder.ipynb  | the main notebook  |
| /camera_cal  | camera calibration images  |
| /images  | screenshots from vidoes |
| /output_images  | images for the writeup |
| /test_images | color, sobel, magnitude filters testing images |
| /utility/util.py | the lib for this project, contains common functions |
| /videos | the testing videos |
| /videos_out | the processed video, only the project_out.mp4 has been uploaded |

### Camera calibration
Before extracting lane lines it is crucial to correct the image distortion that are caused by the camera. Non image-degrading aberrations can easily be corrected using test targets. Samples of chessboard patterns recorded with the same camera that was also used for recording the video are provided in the `camera_cal` folder. 

The code for distortion correction is contained in the code section No.2 of the notebook CarND_Advanced_lane_finder.ipynb.  

We start by preparing "object points", which are (x, y, z) coordinates of the chessboard corners in the world (assuming coordinates such that z=0).  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

`objpoints` and `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistort][image1]
    Distortion-corrected chessboard image

There's also a comparison of the raw road image and the undistorted road image:

![Undistort][image2]
    Distortion-corrected road image

### Thresholded binary image.
From section six to section fifteen I explored the color filter, sobel filter, magnitude filter, and direction filter.

Before I start doing threshold filtering I applied gaussian blurring to the image to reduce the noise. The function is as below:
```python
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
In the exploration of the parameters I set the kernel size to 15 then I found out this is too large for weak lane lines scenarios. It will reduce the capacity of capturing smaller lane lines spots. So I reduced the kernel size to 3 in the video processing pipeline.

Then I explored the HLS channels of a image to understand the features of these different channels.

![HLS channels][image3] 
    HLS channels

From the three channel images we can found out the left yellow lane line seems more obvious in the S channel while the right white lane line is more out standing in the L channel. It seems like a good idea to combine the results from both of the S and L channels but my experiment failed to prove its effectiveness. The L channel is very sensitive to the lighting. A well tuned L filter will completely fail for other scenarios. So I only used H and S channel for color filtering and use other method to pick up the weak lines. 

Then I trialled sobel filter, I only used X direction, magnitude filter and direction filter. I tried different combination of these filters and found out the best one is a mixture of all of them. In the following images it looks like combining color filter and sobelX looks even better but it doesn't work very well in video processing.

![combinedbinary][image4] 
    Combined binary

### Perspective transformation

The next step is doing perspective transformation. I picked up the points from the normal image and use cv2 warp function to convert the image to its bird-eye view. There's also a inverse matrix been generated for pave down the images onto the road view later.

```python
    src = np.float32([[565, 470],[720, 470],[290,680],[1090,680]])
    dst = np.float32([[290, 100],[1090, 100],[290,680],[1090,680]])
    Minv = cv2.getPerspectiveTransform(dst, src)
```

The transformed images are as following:

![birdsEyeView][image6] 
    Birds-eye View

### Detect lane pixels and fit to find the lane boundary

Then I can detect lane lines from the bird-eye view image. 

Firstly I generate a histogram chart of the pixel values of the lower half image accross the x dimension. Then I can use the peaks of left hand side and right hand side image to decide the rough position of the left and right lane lines:

![histogram][image5] 
    Histogram

Secondly, I divide the whole image into 9 horizontal slices and calculate historgram of each slice to find out the position of the left and right lane lines. Extend from the detected line position to its left and right by 100 pixels then we got a filtering window along the detected lane lines:

![slidingWindow][image7] 
    Sliding window

Thirdly, I use the filtering window to sift out valid points along the lane lines and used a second ordered line to fit the points:
```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```
The generated result is like this:

![laneFinder][image8] 
    Lane finder

Please note that in the above scenario there's only one dot in the right hand lane line which is insufficient to do produce a good fit. In this case, I just assume the two lane lines have similar curvature and copy the fit function from the left lane line.
### Determine the curvature of the lane and vehicle position with respect to center

The code section 24 and 25 are calculating the curvature of the lane lines and convert them into metres.

The formula for calculate the curvature is as the following which use first and second derivative to calculate the curvature:

```python
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```
Then I assume the pixel distance can be mapped to distance in real world. My mapping parameters are:

```
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

```

### Warp the detected lane boundaries back onto the original image

Then I used the inverse perspective transformation matrix Minv to pave down the detected lane lines to the original road images. The generated image is like the following:

![pavedBack][image9]
    Paved down lane lines

### Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

The pipeline of the video processing is in code section 27, which is a summary of the experimented methods. Some of the methods were moved to a utility lib located in /utility/util.py. The driving code is in code section 28.

The video processing is simply just takes each frame as a separated image and do processing use the tried techniques. In the generated videos I also output combined binary image, bird-eye view images, curvature or the lane lines and the position of the car.

During processing videos I found out the model can works pretty well in some scenarios but works weirdly in some other scenarios. To investigate the reason I saved video frames into images into the folder /images. Because of the size limit I only uploaded 99 images for each video. I then use these images to tune the parameters. 

I noticed no matter how hard to tune some scenarios are still hard to deal with. Especially the right hand lane lines which is normally very sparse and very easily impacted by shade. To make the model works better I tried some trick to compensate the failed lane lines:

* after have detected the histogram peaks as the lane line bases for the right and left lane lines, I check the value of the histogram in the two lane lines and take the higher value one as the primary side, which means we are confident this side has detected a more reliable lane line. 
* if the value of the primary side is less than 5000, I'll assume this detection is too weak so I'll discard this detection and use the lane lines formula of the previous frame. 
* if the value of the primary side is more than 5000 but the non-primary side lane line peak is less than 2000 then I assume the non-primary side is not getting enough points to fit a lane line so that I'll copy the curv formula from the primary side.
* when copping curv formula, if the non-primary side is too weak to even detect a reasonable lane line base, then I'll use the default lane line distance 750 pixels instead.

Here's a [link to my video result](./videos_out/project_video.mp4)

### Discussion
This project involves lot of parameter tuning, trial and error. There're load of idea to try but limited by time I only tried a few of them. During the project I firstly selected a image as the target and tuned all the parameter to make it works better on the chosen image. But I quickly found out this doesn't guarantee the settings can also be the optimum for other scenarios. Just for example, the sliding window method doesn't work well when there're more than one lines close to each other; the color threshold can't pick up the lines if the lighting has changed or if the lane line color is different; and the model has problem dealing with very curly lanes and the uphill, downhill road. To make the model more robust, I use the difficult scenario images to further tune the parameters. I found out it can help improve the model but still, it is very hard to generalise the model to cover all scenarios.

I'm also curious about how these optical detected lane lines can be used in a real self driving car. There're many many situations in which lane line detection is not useful, for example, when the lanes are merge or branching, or when the lane lines are worn off, incorrect or even no lane lines at all. 
