from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
source_window = 'Image'
maxTrackbar = 10000
rng.seed(12345)
def goodFeaturesToTrack_Demo(val):
    maxCorners = max(val, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.009
    minDistance = 9
    blockSize = 4
    gradientSize = 3
    useHarrisDetector = True
    k = 0.04
    # Copy the source image
    copy = np.copy(src)
    # Apply corner detection
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, useHarrisDetector=useHarrisDetector, k=k)

    print(corners)
    # Draw corners detected
    print('** Number of corners detected:', corners.shape[0])
    radius = 3
    for i in range(corners.shape[0]):
        cv.circle(copy, (corners[i,0,0], corners[i,0,1]), radius, (0,0,255) , cv.FILLED)
    # Show what you got
    cv.namedWindow(source_window)
    cv.imshow(source_window, copy)
    # Set the needed parameters to find the refined corners
    winSize = (5, 5)
    zeroZone = (1, 1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    # Calculate the refined corner locations
    corners = cv.cornerSubPix(src_gray, corners, winSize, zeroZone, criteria)
    # Write them down
    for i in range(corners.shape[0]):
        print(" -- Refined Corner [", i, "]  (", corners[i,0,0], ",", corners[i,0,1], ")")


#BLURRING AND UNSHARPENING 
image = cv.imread("/home/ale/Scrivania/3sh/PROVARE/erode22_gaussian33.bmp")
gaussian_3 = cv.GaussianBlur(image, (3,3), 10.0)
unsharp_image = cv.addWeighted(image, 1.6, gaussian_3, -0.5, 0, image)
#cv2.imwrite("unsharpened", unsharp_image)


# Load source image and convert it to gray
parser = argparse.ArgumentParser(description='Code for Shi-Tomasi corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='/home/ale/Scrivania/3sh/PROVARE/erode22_gaussian33.bmp')
args = parser.parse_args()
#src = cv.imread('/home/ale/Scrivania/3sh/PROVARE/eroded.bmp')
src=unsharp_image
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Create a window and a trackbar
cv.namedWindow(source_window)
maxCorners = 10 # initial threshold
cv.createTrackbar('Threshold: ', source_window, maxCorners, maxTrackbar, goodFeaturesToTrack_Demo)
cv.imshow(source_window, src)
goodFeaturesToTrack_Demo(maxCorners)
cv.waitKey()
