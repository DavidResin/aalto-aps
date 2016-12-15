import numpy as np
import cv2

cap = cv2.VideoCapture('people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while (1):
	ret, frame = cap.read()

	fgmask = fgbg.apply(frame)

	kernel = np.ones((2,2), np.uint8)
	erosion = cv2.erode(fgmask, kernel, iterations=2)

	cv2.imshow('fgmask', frame)
	cv2.imshow('frame', fgmask)
	cv2.imshow('erosion', erosion)

	k = cv2.waitKey(30) & 0xFF
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()