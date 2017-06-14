import cv2, imutils
import numpy as np
from matplotlib import pyplot as plt

def outerEdge(image):
	kernel = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]])
	temp1 = cv2.filter2D(image, -1, kernel)
	temp2 = cv2.bitwise_and(temp1, cv2.bitwise_not(image))
	ret, border = cv2.threshold(temp2, 10, 255, cv2.THRESH_BINARY)
	return border

def innerEdge(image):
	reverse = cv2.bitwise_not(image)
	return outerEdge(reverse)

def expand(image, index):
	temp = np.zeros(image.shape, dtype=np.uint8)
	temp[image == index] = 255
	temp = outerEdge(temp)
	image[temp == 255] = index

def distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2

	return (x1 - x2)**2 + (y1 - y2)**2

def watershed(image, mask, edges, center1, center2):
	kernel = np.ones((3, 3), np.uint8)
	imageColor = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)


	ret, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
	sure_bg = cv2.dilate(opening, kernel, iterations=3)
	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
	sure_fg = cv2.erode(sure_fg, kernel, iterations=1)
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg, sure_fg)

	ret, markers = cv2.connectedComponents(sure_fg)
	markers = markers + 1
	markers[unknown == 255] = 0
	temp = np.copy(markers)
	markers = cv2.watershed(imageColor, markers)
	markers[mask == 0] = -1
	'''
	plt.subplot(2, 3, 1), plt.imshow(image)
	plt.subplot(2, 3, 2), plt.imshow(opening)
	plt.subplot(2, 3, 3), plt.imshow(dist_transform)
	plt.subplot(2, 3, 4), plt.imshow(sure_fg)
	plt.subplot(2, 3, 5), plt.imshow(temp)
	plt.subplot(2, 3, 6), plt.imshow(markers)
	plt.tight_layout()
	plt.show()
	'''
	cut_mask = np.zeros(mask.shape, dtype=np.uint8)
	cut_mask[markers == 0] = 255

	dots = []
	i = 0

	while dots or i == 0:
		if dots:
			count, sumX, sumY = 0, 0, 0
		
			for (x, y) in dots:
				count += 1
				sumX += x
				sumY += y

			expand(markers, i)
			center = (int(round(sumX / count)), int(round(sumY / count)))
			
			if distance(center, center1) < distance(center, center2):
				cut_mask[markers == i] = 255

		i += 1
		dots = list(zip(*np.where(markers == i)))

	return cv2.bitwise_and(mask, cut_mask), cv2.bitwise_and(mask, cv2.bitwise_not(cut_mask))

def center(mask1, mask2):
	points1 = list(zip(*np.where(cv2.bitwise_and(mask1, outerEdge(mask2)) == 255)))
	center1 = tuple(map(lambda y: int(round(sum(y) / float(len(y)))), zip(*points1)))
	points2 = list(zip(*np.where(cv2.bitwise_and(mask2, outerEdge(mask1)) == 255)))
	center2 = tuple(map(lambda y: int(round(sum(y) / float(len(y)))), zip(*points2)))
	return center1, center2

def cut(image1, image2, mask1, mask2):
	kernel = np.ones((5, 5), np.float32) / 25
	gray1 = cv2.filter2D(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), -1, kernel)
	gray2 = cv2.filter2D(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), -1, kernel)

	center1, center2 = center(mask1, mask2)

	mask = cv2.bitwise_and(mask1, mask2)
	diff = cv2.absdiff(gray1, gray2)
	diff = cv2.bitwise_and(mask, diff)
	edges = cv2.bitwise_xor(mask1, mask2)

	c1, c2 = watershed(diff, mask, edges, center1, center2)
	'''
	a = imutils.resize(c1, width=1000)
	cv2.imshow("test", a)
	cv2.waitKey(0)

	a = imutils.resize(c2, width=1000)
	cv2.imshow("test", a)
	cv2.waitKey(0)
	'''
	return c1, c2