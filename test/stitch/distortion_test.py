import argparse, cv2, glob, imutils, re, sys, math
import numpy as np

def lensCorrect(x, y, halfW, halfH, cR):
	cX = x - halfW
	cY = y - halfH
	ratio = math.sqrt(cX**2 + cY**2) * cR

	if ratio == 0:
		theta = 1;
	else:
		theta = math.atan(ratio) / ratio

	newX = halfW + theta * cX
	newY = halfH + theta * cY

	return (int(newX), int(newY))

def paddings(w, h, cR):
	padX, _ = lensCorrect(0, h / 2, w / 2, h / 2, cR)
	_, padY = lensCorrect(w / 2, 0, w / 2, h / 2, cR)

	return (-int(padX), -int(padY))

zoom = 1.28
image_orig = cv2.imread("distortion_test.jpg")
h_orig, w_orig, ch = image_orig.shape
image = cv2.resize(image_orig, (int(zoom*w_orig), int(zoom*h_orig)))
h, w, ch = image.shape
halfW = w / 2
halfH = h / 2

strength = 2
correctionRadius = strength / math.sqrt(w**2 + h**2)

padX, padY = paddings(w, h, correctionRadius)

newW = w + 2 * padX
newH = h + 2 * padY

newImage = np.zeros((newH, newW, ch), np.uint8)

padXb = int((newW - w_orig) / 2)
padYb = int((newH - h_orig) / 2)

for x in range(w):
	for y in range(h):
		newX, newY = lensCorrect(x, y, halfW, halfH, correctionRadius)
		newImage[padY + newY][padX + newX] = image[y][x]

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

cv2.imwrite("distortion_result_forwards.jpg", newImage)