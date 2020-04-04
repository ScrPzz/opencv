import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt 

#parameters
kernel_size=5
low_thresh = 40
high_thresh = 80

rho = 0.6  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 28  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 21  # maximum gap in pixels between connectable line segments




###Loadign the image, making it gray and blurring
img=cv.imread('/home/ale/Scrivania/3sh/PROVARE/erode22_gaussian33.bmp')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur_gray= cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

### Finding edges with canny

edges = cv.Canny(blur_gray, low_thresh, high_thresh, apertureSize = 3)

### Hough probabilistic lines detector

line_image = np.copy(img) * 0  # creating a blank to draw lines on

lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

a,b,c = lines.shape
for i in range(a):
    cv.line(line_image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 200, 255), 1, cv.LINE_AA)
lines_edges = cv.addWeighted(img, 0.7, line_image, 1, 0)


lines_edges = cv.addWeighted(img, 0.7, line_image, 1, 0)

cv.imshow("Lines", lines_edges)
cv.imwrite("Lines.png",lines_edges)

### Hough not p lines detector
lines_notp = cv.HoughLines(edges,1,np.pi/180,200)

for rho,theta in lines_notp[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

lines_edges = cv.addWeighted(img, 0.7, line_image, 1, 3)

cv.imshow("Lines_notp", lines_edges)
cv.imwrite("Lines_notp.png",lines_edges)


cv.waitKey(0)



