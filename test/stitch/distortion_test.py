import argparse, cv2, glob, imutils, re, sys, math
import numpy as np

def factor(x, y, cR):
	ratio = math.sqrt(x**2 + y**2) * cR

	if ratio == 0:
		theta = 1;
	else:
		theta = math.atan(ratio) / ratio

	return theta

def lensCorrect(x, y, halfW, halfH, cR, theta_inv, zoomX, zoomY):
	cX = x - halfW
	cY = y - halfH

	theta = factor(int(cX * zoomX), int(cY * zoomY), cR)

	newX = (halfW + theta * cX * theta_inv) * zoomX
	newY = (halfH + theta * cY * theta_inv) * zoomY

	return (int(newX), int(newY))

def paddings(w, h, cR):
	theta_inv = 1 / factor(w / 2, h / 2, cR)
	padX, _ = lensCorrect(0, h / 2, w / 2, h / 2, cR, theta_inv, 1, 1)
	_, padY = lensCorrect(w / 2, 0, w / 2, h / 2, cR, theta_inv, 1, 1)

	return (-int(padX), -int(padY), theta_inv)

zoom = 1
image_orig = cv2.imread("distortion_test.jpg")
h_orig, w_orig, ch = image_orig.shape
image = cv2.resize(image_orig, (int(zoom * w_orig), int(zoom * h_orig)))
h, w, ch = image.shape
halfW = w / 2
halfH = h / 2

strength = 3
correctionRadius = strength / math.sqrt(w**2 + h**2)

padX, padY, theta_inv = paddings(w, h, correctionRadius)

newW = w + 2 * padX
newH = h + 2 * padY

newHalfW = newW / 2
newHalfH = newH / 2

zoomX = newW / w
zoomY = newH / h
antiZoomX = 1 / zoomX
antiZoomY = 1 / zoomY

image = cv2.resize(image, (newW, newH))

newImage = np.zeros((newH, newW, ch), np.uint8)

padXb = int((newW - w_orig) / 2)
padYb = int((newH - h_orig) / 2)

for x in range(newW):
	for y in range(newH):
		newX, newY = lensCorrect(x, y, newHalfW, newHalfH, correctionRadius, theta_inv, antiZoomX, antiZoomY)
		newImage[padY + newY][padX + newX] = image[y][x]
		if x + 2 < newW:
			newImage[padY + newY][padX + newX + 1] = image[y][x]
		if y + 2 < newH:
			newImage[padY + newY + 1][padX + newX] = image[y][x]
		if x + 2 < newW and y + 2 < newH:
			newImage[padY + newY + 1][padX + newX + 1] = image[y][x]

for x in range(w_orig):
	newImage[padYb][padXb + x] = [255, 255, 255]
	newImage[padYb + h_orig-1][padXb + x] = [255, 255, 255]

for y in range(h_orig):
	newImage[padYb + y][padXb] = [255, 255, 255]
	newImage[padYb + y][padXb + w_orig-1] = [255, 255, 255]

for x in range(newW):
	newImage[0][x] = [255, 255, 255]
	newImage[newH-1][x] = [255, 255, 255]

for y in range(newH):
	newImage[y][0] = [255, 255, 255]
	newImage[y][newW-1] = [255, 255, 255]

cv2.imwrite("distortion_result" + str(strength) + ".jpg", newImage)