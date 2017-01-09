import numpy as np
import imutils
import cv2

img = cv2.imread("panorama/coins.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((2, 2), np.uint8)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

bg = cv2.dilate(closing, kernel, iterations=3)

dist_transform = cv2.distanceTransform(bg, cv2.DIST_L2, 3)
ret, fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

fg = np.uint8(fg)
uk = cv2.subtract(bg, fg)

cv2.imshow("thresh", thresh)
cv2.imshow("closing", closing)
cv2.imshow("dist_transform", dist_transform)
cv2.imshow("bg", bg)
cv2.imshow("fg", fg)
cv2.imshow("uk", uk)
cv2.waitKey(0)