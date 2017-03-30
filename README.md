**Advanced Lane Finding Project**

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

[video1]: ./videos_out/project_video.mp4 "Video processed"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
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
Before extracting lane lines it is crucial to correct the image distortion that are caused by the camera. Non image-degrading abberations such as pincussion/barrel distortion can easily be corrected using test targets. Samples of chessboard patterns recorded with the same camera that was also used for recording the video are provided in the `camera_cal` folder. 

The code for distortion correction is contained in the code section No.2 of the notebook CarND_Advanced_lane_finder.ipynb.  

We start by preparing "object points", which are (x, y, z) coordinates of the chessboard corners in the world (assuming coordinates such that z=0).  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

`objpoints` and `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![Undistort][image1]### Distortion-corrected chessboard image
There's also a comparision of the raw road image and the undistorted road image:
![Undistort][image2]### Distortion-corrected road image

### Thresholded binary image.
Before I start doing threshold filtering I applied gaussian blurring to the image to reduce the noise. The function is as below:
```python
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
In the exploration of the parameters I set the kernal size to 15 but found out this is so large that it will reduce the capacity to capture spotty lane lines. So I reduced the kernel size to 3 in the video processing pipeline.

Then I explored the HLS channels of a image to upderstand the features of these different channels.
![HLS channels][image3]### HLS channels



### Perspective transformation
### Detect lane pixels and fit to find the lane boundary
### Determine the curvature of the lane and vehicle position with respect to center
### Warp the detected lane boundaries back onto the original image
### Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
### Discussion
This project involves lot of parameter tuning, trial and error. There're load of idea to try but limited by time I only tried a few of them. During the project I firstly selected a image as the target and tuned all the parameter to make it works better on the chosen image. But I quickly found out this doesn't garentee the settings can also be the optimum for other scenarios. Just for example, the sliding window method doesn't work well when there're more than one lines close to each other; the color threshold can't pick up the lines if the lighting has changed or if the lane line color is different; and the model has problem dealing with very curly lanes and the uphill, downhill road. To make the model more robust, I use the difficult scenario images to further tune the parameters. I found out it can help improve the model but still, it is very hard to generalize the model to cover all scenarios.

I'm also curious about how these optical detected lane lines can be used in a real self driving car. There're many many situations in which lane line detection is not useful, for example, when the lanes are merge or branching, or when the lane lines are worn off, incorrect or even no lane lines at all. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Apply a distortion correction to raw images.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

