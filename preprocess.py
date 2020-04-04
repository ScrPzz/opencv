from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import os
from scipy.ndimage.filters import median_filter

#CREATING FOLDER FOR RESULTS

dirname = 'sharpening_results'
os.mkdir(dirname)

os.chdir(dirname)
#KERNEL SHARPENING

image = cv.imread('/home/ale/Scrivania/3sh/Texp_90_134356_620_1_CUT.bmp')

cv.imshow('Original', image)

# Create shapening kernel,
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 8,-1],
                              [-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
sharpened1 = cv.filter2D(image, -1, kernel_sharpening)
cv.imshow('Image Sharpening', sharpened1)
cv.imwrite('kernel_sharpened.bmp', sharpened1)

#KERNEL GAUSSIAN BLUR SHARPENING
gaussian_3 = cv.GaussianBlur(image, (9,9), 10.0)
unsharp_image = cv.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 8,-1],
                              [-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
sharpened2 = cv.filter2D(unsharp_image, -1, kernel_sharpening)
cv.imshow('kernerl_blur_sharpened', sharpened2)
cv.imwrite('kernel__blursharpened.bmp', sharpened2)


#GAUSSIAN BLUR MASK
image = cv.imread("/home/ale/Scrivania/3sh/Texp_90_134356_620_1_CUT.bmp")
gaussian_3 = cv.GaussianBlur(image, (9,9), 10.0)
sharpened3 = cv.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
cv.imshow('gaussian_sharpened', sharpened3)
cv.imwrite("gaussian_sharpened.bmp", sharpened3)



#ERODE
image = cv.imread("/home/ale/Scrivania/3sh/Texp_90_134356_620_1_CUT.bmp")
kernel22 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
kernel33 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
kernel44 = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
kernel55 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

dlt22= cv.erode(image, kernel= kernel22, iterations =1)
dlt33= cv.erode(image, kernel= kernel33, iterations =1)
dlt44= cv.erode(image, kernel= kernel44, iterations =1)
dlt55= cv.erode(image, kernel= kernel55, iterations =1)


cv.imwrite('eroded22.bmp', dlt22)
cv.imwrite('eroded33.bmp', dlt33)
cv.imwrite('eroded44.bmp', dlt33)
cv.imwrite('eroded55.bmp', dlt55)


#ERODE AND GAUSSIAN SHARP

image = cv.imread("/home/ale/Scrivania/3sh/Texp_90_134356_620_1_CUT.bmp")


gaussian_3_erode2 = cv.GaussianBlur(dlt22, (3,3), 0)
gaussian_3_erode3 = cv.GaussianBlur(dlt33, (3,3), 0)
gaussian_3_erode4 = cv.GaussianBlur(dlt44, (3,3), 0)
erode3_sharp2 = cv.addWeighted(dlt22, 1.5, gaussian_3_erode2, -0.5, 0, image)
erode3_sharp3 = cv.addWeighted(dlt33, 1.5, gaussian_3_erode3, -0.5, 0, image)
erode3_sharp4 = cv.addWeighted(dlt44, 1.5, gaussian_3_erode4, -0.5, 0, image)

#cv.imshow('eroded_gaussian', erode_sharp)
cv.imwrite('erode_gaussian.bmp', erode3_sharp2)
cv.imwrite('erode_gaussian.bmp', erode3_sharp3)
cv.imwrite('erode_gaussian.bmp', erode3_sharp4)


#ERODE LAPLACIAN 

image = cv.imread("/home/ale/Scrivania/3sh/Texp_90_134356_620_1_CUT.bmp")

gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)

lap = cv.Laplacian(gray, cv.CV_64F, ksize=3)

erode_laplace_sharp = gray - 0.7*lap

cv.imshow('erode_laplace', erode_laplace_sharp)

cv.imwrite('erode_laplace.bmp', erode_laplace_sharp)


#TRESHOLDING 

img = cv.imread("/home/ale/Scrivania/3sh/Texp_90_134356_620_1_CUT.bmp")
_, threshold = cv.threshold(img, 155, 255, cv.THRESH_BINARY)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gaus = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 12)

cv.imshow("Binary threshold", threshold)
cv.imwrite("Binary threshold.bmp", threshold)
cv.imshow("Gaussian", gaus)
cv.imwrite("gauss_thresh.bmp", gaus)



#ERODE THRESHOLD 
img = cv.imread("/home/ale/Scrivania/3sh/Texp_90_134356_620_1_CUT.bmp")



dlt22= cv.erode(image, kernel= kernel22, iterations =1)

_, threshold22 = cv.threshold(dlt22, 100, 255, cv.THRESH_BINARY)

img_gray22 = cv.cvtColor(dlt22, cv.COLOR_BGR2GRAY)

gaus22 = cv.adaptiveThreshold(img_gray22, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 12)



dlt33= cv.erode(image, kernel= kernel33, iterations =1)

_, threshold33 = cv.threshold(dlt33, 100, 255, cv.THRESH_BINARY)

img_gray33 = cv.cvtColor(dlt33, cv.COLOR_BGR2GRAY)

gaus33 = cv.adaptiveThreshold(img_gray33, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 12)



dlt44= cv.erode(image, kernel= kernel44, iterations =1)

_, threshold44 = cv.threshold(dlt44, 100, 255, cv.THRESH_BINARY)

img_gray44 = cv.cvtColor(dlt44, cv.COLOR_BGR2GRAY)

gaus44 = cv.adaptiveThreshold(img_gray44, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 91, 12)



cv.imshow("Binary threshold + erode 22", threshold22)
cv.imwrite("Binary threshold_erode_33.bmp", threshold22)
cv.imshow("Gaussian erode_22", gaus22)
cv.imwrite("Gaussian_erode_22.bmp", gaus22)


cv.imshow("Binary threshold + erode 33", threshold33)
cv.imwrite("Binary threshold_erode_33.bmp", threshold33)
cv.imshow("Gaussian erode_33", gaus33)
cv.imwrite("Gaussian_erode_33.bmp", gaus33)

cv.imshow("Binary threshold + erode 44", threshold44)
cv.imwrite("Binary threshold_erode_44.bmp", threshold44)
cv.imshow("Gaussian erode 44", gaus44)
cv.imwrite("Gaussian_erode_44.bmp", gaus44)


if cv.waitKey(0) & 0xff == 27 :
    cv.destroyAllWindows()


