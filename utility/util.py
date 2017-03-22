import numpy as np
import cv2
import matplotlib.pyplot as plt

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def pltPairedShow(img1,title1, img2, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def pltShowFour(img1,title1, img2, title2, img3, title3, img4, title4):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax2.imshow(img2,cmap ='gray')
    ax2.set_title(title2)
    ax3.imshow(img3,cmap ='gray')
    ax3.set_title(title3)
    ax4.imshow(img4,cmap ='gray')
    ax4.set_title(title4)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
#def hls_color_thresh(img, threshH,threshL, threshS):
def hls_s_thresh(img, Sthresh):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    binary_output = np.zeros_like(img)
    binary_output[(imgHLS[:,:,2] >= Sthresh[0])  & (imgHLS[:,:,2] <= Sthresh[1])] = 255
    return binary_output

def sobelX_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):   
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    sobelx = cv2.Sobel(imgHLS[:,:,2], cv2.CV_64F, 1,0, ksize=sobel_kernel)
    scaled_sobel = np.uint8(255*sobelx / np.max(sobelx))
       
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255

    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):    
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    gray = imgHLS[:,:,2]
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*gradmag / np.max(gradmag))
       
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255


    # 6) Return this mask as your binary_output image
    return binary_output

#Direction threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    gray = imgHLS[:,:,2]
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    absgraddir = np.arctan2(abs_sobely, abs_sobelx) 

    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    return binary_output

#Both Magnitude and direction threshold
def mag_dir_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0,np.pi/2)):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    gray = imgHLS[:,:,2]
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize=sobel_kernel) 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    absgraddir = np.arctan2(abs_sobely, abs_sobelx) 

    scaled_sobel = np.uint8(255*gradmag / np.max(gradmag))
       
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1]) & (absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1]) ] = 255
    
    return binary_output