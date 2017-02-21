#importing some useful packages
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    xsize = img.shape[1]
    ysize = img.shape[0]
    horizont = int(0.6*ysize)
    avgSlope1 = 1
    avgSlope2 = 1
    avgx1 = 0
    avgx2 = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (x1-x2)/(y1-y2)
            if(slope.item() > 0.01 and slope.item() < 2.0):
                if (avgSlope1==0):
                    avgSlope1 = avgx1
                else:
                    avgSlope1 = (avgSlope1 + slope)/2
                print('R: ', slope, '\t\t', avgSlope1)
                x1Aux = int(x2 - avgSlope1*(y2-ysize))
                if (avgx1==0):
                    avgx1 = x1Aux
                else:
                    avgx1 = (avgx1 + x1Aux)/2
                x2Aux = int(x1Aux + avgSlope1*(y2-ysize))
                cv2.line(img, (x1Aux, ysize), (x2Aux, horizont), [0, 255, 255], thickness)                
            if(slope.item() > -2.0 and slope.item() < -0.01):
                avgSlope2 = (avgSlope2 + slope)/2
                print('L: ', slope, '\t\t', avgSlope2)
                x1Aux = int(x2 -avgSlope2*(y2-ysize))
                x2Aux = int(x1Aux + avgSlope2*(y2-ysize))
                cv2.line(img, (x1Aux, ysize), (x2Aux, horizont), [255, 0, 0], thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    

import os
test_images = os.listdir("test_images/")

for image in test_images:
    image = mpimg.imread('test_images/' + image, cv2.COLOR_RGB2GRAY)
    #image = mpimg.imread('test_images/' + 'challenge_3.png', cv2.COLOR_RGB2GRAY)
    #Get the size of the image
    xsize=image.shape[1]
    ysize=image.shape[0]
    print(xsize, " ", ysize)
    grayimage = grayscale(image)
    edgy = gaussian_blur(grayimage, 3)
    low_threshold = 30
    high_threshold = 30*4.4
    edgy_image = canny(edgy, low_threshold, high_threshold)
    plt.imshow(edgy_image, cmap='gray')
    plt.pause(0.1)
#    tin = input("Test Input: ")

    left_down_corner = 0.05*xsize
    rigth_down_corner = 0.95*xsize
    horizont = 0.6*ysize
    horizont_margin = 0.07*xsize
    middle = xsize/2
    
    vertices = np.array([[(left_down_corner,ysize), 
                  (middle-horizont_margin, horizont), 
                  (middle + horizont_margin, horizont), 
                  (rigth_down_corner, ysize) ]], dtype=np.int32)
    image_interest = region_of_interest(edgy_image, vertices)
    plt.imshow(image_interest)
    rho = 3
    theta = np.pi/1000
    threshold = 50
    min_line_len = 4
    max_line_gap = 10
    img_lines = hough_lines(image_interest, rho, theta, threshold, min_line_len, max_line_gap)
    final_img = weighted_img(img_lines, image)
    plt.imshow(final_img)
    plt.pause(0.01)
    tin = input("Test Input: ")