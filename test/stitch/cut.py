import argparse
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt

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
ret, mask1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
ret, mask2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_and(mask1, mask2)

#get blurred inverted image
diff = cv2.bitwise_and(diff, mask)
diff = cv2.GaussianBlur(diff, (5, 5), 1.4)
diffColor = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

ret, thresh = cv2.threshold(diff, 220, 255, cv2.THRESH_BINARY)

########################################################################

kernel = np.ones((3, 3),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations=2)

bg = cv2.dilate(closing,kernel,iterations=3)

dist_transform = cv2.distanceTransform(bg,cv2.DIST_L2,3)
ret, fg = cv2.threshold(dist_transform, .7 * dist_transform.max(), 255, 0)

fg = np.uint8(fg)
uk = cv2.subtract(bg, fg)

ret, markers = cv2.connectedComponents(fg)
markers = markers + 1

markers[uk == 255] = 0

##############################################################################

#run watershed algorithm
markers = cv2.watershed(diffColor, markers)

#draw the lines on the difference image
diffColor[markers == -1] = [255, 0, 0]

plt.subplot(4, 4, 1),plt.imshow(img1)
plt.title('Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 2),plt.imshow(img2)
plt.title('Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 3),plt.imshow(thresh, cmap='gray')
plt.title('Threshold'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 4),plt.imshow(closing, cmap='gray')
plt.title('Closing'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 5),plt.imshow(gray1, cmap='gray')
plt.title('Gray 1'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 6),plt.imshow(gray2, cmap='gray')
plt.title('Gray 2'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 7),plt.imshow(bg, cmap='gray')
plt.title('Background'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 8),plt.imshow(fg, cmap='gray')
plt.title('Foreground'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 9),plt.imshow(diff, cmap='gray')
plt.title('Difference'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 10),plt.imshow(mask, cmap='gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 11),plt.imshow(uk, cmap='gray')
plt.title('Unknown'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 12),plt.imshow(dist_transform, cmap='gray')
plt.title('Distance Transform'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 13),plt.imshow(mask1, cmap='gray')
plt.title('Mask 1'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 14),plt.imshow(mask2, cmap='gray')
plt.title('Mask 2'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 15),plt.imshow(markers)
plt.title('Markers'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 4, 16),plt.imshow(diffColor)
plt.title('Result'), plt.xticks([]), plt.yticks([])
'''

plt.subplot(4, 2, 1),plt.imshow(img1, cmap="rainbow")
plt.title('Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 2, 2),plt.imshow(img2)
plt.title('Image 2'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 3),plt.imshow(diff, cmap='gray')
plt.title('Difference'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 2, 4),plt.imshow(dist_transform, cmap='gray')
plt.title('Distance Transform'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 5),plt.imshow(bg)
plt.title('bg'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 2, 6),plt.imshow(fg)
plt.title('fg'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 7),plt.imshow(markers)
plt.title('Markers'), plt.xticks([]), plt.yticks([])
plt.subplot(4, 2, 8),plt.imshow(diffColor)
plt.title('Result'), plt.xticks([]), plt.yticks([])
'''
plt.tight_layout()
plt.show()