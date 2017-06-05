import cv2
import numpy as np
from matplotlib import pyplot as plt

def vanilla_watershed(image, mask):
	image = cv2.bitwise_or(image, cv2.bitwise_not(mask))
	img = cv2.imread('coins.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	gray2 = image
	ret, thresh2 = cv2.threshold(gray2, 1,255,cv2.THRESH_BINARY)
	thresh = thresh2
	img = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
	print(img.dtype)
	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=3)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.0001*dist_transform.max(), 255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1

	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0

	markers = cv2.watershed(img,markers)

	img[markers == -1] = [255,0,0]

	cv2.imshow("test", img)
	cv2.waitKey(0)
	plt.imshow(dist_transform)
	plt.show()




def watershed(image, mask):
	shape = image.shape[:2]
	flood = np.copy(image)
	labels = np.zeros(shape, dtype=np.int32)
	covered = np.zeros(shape, dtype=np.uint8)
	i = 0

	for level in range(32):
		level = min(8 * level, 255)
		print(level)
		_, flood = cv2.threshold(flood, level, 255, cv2.THRESH_TOZERO)
		pos = list(set(zip(*np.where(flood == 0))) & set(zip(*np.where((mask == 255)))))
		empty = [p for p in pos if labels[p] == 0]

		spread(empty, labels, flood, covered)

		i = spawn(shape, empty, labels, flood, covered, i)

		cv2.imshow("test", image)
		cv2.waitKey(0)
		cv2.imshow("test", flood)
		cv2.waitKey(0)
		cv2.imshow("test", covered)
		cv2.waitKey(0)
		cv2.imshow("test", labels)
		cv2.waitKey(0)

def spread(points, labels, flood, covered):
	edges = outerEdge(covered)
	newPoints = list(set(zip(*np.where(edges == 255))) & set(points))

	while newPoints:
		p = newPoints.pop(0)
		new, old = neighbors(p, newPoints)

		old = [o for o in old if covered[o] == 255 and labels[o] != -1]

		labels[p] = labels[old[0]]

		for o in old[1:]:
			if labels[old[0]] != labels[o]:
				labels[p] = -1
				
		newPoints.append(new)
		covered[p] = 255
		flood[p] = 255

def spawn(shape, points, labels, flood, covered, i):
	while points:
		neighbors = [points[0]]

		while neighbors:
			p = neighbors[0]
			points.remove(p)
			labels[p] = i
			flood[p] = 255
			covered[p] = 255
			print(p, points)
			n, _ = neighbors(p, points)
			neighbors.append(n)

		i += 1

	return i

def neighbors(pos, elements):
	x, y = pos
	candidates = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]

	good, bad = [], []

	for c in candidates:
		if c in elements:
			good.append(c)
		else:
			bad.append(c)

	return good, bad

def outerEdge(image):
	kernel = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]])
	temp1 = cv2.filter2D(image, -1, kernel)
	temp2 = cv2.bitwise_and(temp1, cv2.bitwise_not(image))
	ret, border = cv2.threshold(temp2, 1, 255, cv2.THRESH_BINARY)
	return border

image1 = cv2.imread("test1.jpg")
image2 = cv2.imread("test2.jpg")
mask1 = cv2.imread("mask1.jpg")
mask2 = cv2.imread("mask2.jpg")

image1 = image1[350:1050, 1000:2100]
image2 = image2[350:1050, 1000:2100]
mask1 = mask1[350:1050, 1000:2100]
mask2 = mask2[350:1050, 1000:2100]

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

mask = cv2.bitwise_and(mask1, mask2)
diff = cv2.absdiff(gray1, gray2)
#diff = cv2.bitwise_not(diff)
diff = cv2.bitwise_and(mask, diff)
vanilla_watershed(diff, mask)

#cv2.imwrite("diff.jpg", diff)