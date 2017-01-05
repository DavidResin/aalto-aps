import cv2
import numpy as np

img = cv2.imread("home.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(gray, None)
surf.setHessianThreshold(50)

img2 = cv2.drawKeypoints(gray, kp, None, (0, 0, 255), 4)

cv2.imwrite("surf_keypoints.jpg", img2)