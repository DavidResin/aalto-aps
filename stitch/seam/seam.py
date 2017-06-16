import cv2
import numpy as np

def common(image1, image2):
	return cv2.bitwise_and(image1, image2)

def outerEdge(image):
	kernel = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]])
	temp1 = cv2.filter2D(image, cv2.CV_8U, kernel)
	temp2 = cv2.bitwise_and(temp1, cv2.bitwise_not(image))
	ret, border = cv2.threshold(temp2, 1, 255, cv2.THRESH_BINARY)
	return border

def innerEdge(image):
	reverse = cv2.bitwise_not(image)
	return outerEdge(reverse)

def neighbors(pos, image):
	to_process = list(pos)

	while to_process:
		temp = to_process[0]
		to_process.pop(0)
		image[temp] = 255
		to_process += list(set(neighbors_rec(temp, image)) - set(to_process))

def neighbors_rec(pos, image):
	x, y = pos
	candidates = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
	return [i for i in candidates if image[i] == 0]

def zones(image1, image2, cut):
	intersection = common(image1, image2)
	border1 = innerEdge(image1)
	border2 = innerEdge(image2)
	edge1 = common(intersection, border2)
	edge2 = common(intersection, border1)
	zone1 = cv2.bitwise_or(cut, edge1)
	tips = cv2.bitwise_and(cut, edge1)
	firstPoints = cv2.bitwise_and(cv2.bitwise_and(intersection, outerEdge(edge1)), cv2.bitwise_and(cv2.bitwise_not(cut), cv2.bitwise_not(cv2.bitwise_or(tips, outerEdge(tips)))))
	pos = tuple(zip(*np.where(firstPoints == 255)))
	neighbors(pos, zone1)
	zone2 = cv2.bitwise_and(cv2.bitwise_not(zone1), intersection)

	return zone1, zone2




image1 = np.zeros((10, 10, 1), dtype=np.uint8)
image2 = np.zeros((10, 10, 1), dtype=np.uint8)

image1[0:9,0:9] = 255
image2[1:,1:] = 255
'''
cut = np.zeros((400, 400, 1), dtype=np.uint8)
for i in range(100):
	cut[200+i, 299-i] = 255
'''

cv2.imshow("test", image1)
cv2.waitKey(0)
cv2.imshow("test", image2)
cv2.waitKey(0)
cut = np.array(
	[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 255, 0],
	[0, 0, 0, 0, 0, 0, 0, 255, 0, 0],
	[0, 0, 0, 0, 255, 0, 0, 255, 0, 0],
	[0, 0, 0, 255, 0, 255, 0, 0, 255, 0],
	[0, 0, 255, 0, 0, 255, 255, 255, 255, 0],
	[0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
	[0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
	[0, 255, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

intersection = common(image1, image2)
border1 = innerEdge(image1)
border2 = innerEdge(image2)
edge1 = common(intersection, border2)
edge2 = common(intersection, border1)
tips = cv2.bitwise_and(edge2, edge1)
cv2.imshow("test", tips)
cv2.waitKey(0)
'''
cv2.imshow("test", cut)
cv2.waitKey(0)

zone1, zone2 = zones(image1, image2, cut)

total = cv2.bitwise_or(zone1, zone2)

cv2.imshow("test", zone1)
cv2.waitKey(0)
cv2.imshow("test", zone2)
cv2.waitKey(0)

cv2.imshow("test", total)
cv2.waitKey(0)
'''