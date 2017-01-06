import argparse
import imutils
import numpy as np
import cv2

#setup arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="path to the first image")
ap.add_argument("-s", "--second", required=True, help="path to the second image")
args = vars(ap.parse_args())

#read images
img1 = cv2.imread(args["first"])
img2 = cv2.imread(args["second"])

#resize images
img1 = imutils.resize(img1, width=800)
img2 = imutils.resize(img2, width=800)

#extract gray-level images
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#get absolute difference
diff = cv2.absdiff(gray1, gray2)
diff = cv2.bitwise_not(diff)

#create mask for difference
ret, thresh1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_and(thresh1, thresh2)

#get blurred inverted image
diff = cv2.bitwise_and(diff, mask)
ret, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
diff = cv2.GaussianBlur(diff, (5, 5), 1.4)


kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
bg = cv2.dilate(opening, kernel, iterations=3)

#bg = diff
fg = cv2.distanceTransform(diff, cv2.DIST_L2, 5)
ret, fg = cv2.threshold(fg, .7 * fg.max(), 255, 0)

fg = np.uint8(fg)
uk = cv2.subtract(bg, fg)

ret, markers = cv2.connectedComponents(fg)
markers = markers + 1
markers[uk == 255] = 0
markers = cv2.watershed(img1, markers)

img1[markers == -1] = [255, 0, 0]

cv2.imshow("Markers", markers)
cv2.imshow("Image", img1)
cv2.waitKey(0)