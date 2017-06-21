import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
	_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_red = np.array([10, 50, 50])
	upper_red = np.array([50, 255, 255])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(frame, frame, mask=mask)

	kernel = np.ones((15, 15), np.float32) / 225
	kernel2 = np.ones((5, 5), np.uint8)

	#smoothed = cv2.filter2D(res, -1, kernel)
	#gauss = cv2.GaussianBlur(res, (15, 15), 0)
	#median = cv2.medianBlur(res, 15)
	#bilateral = cv2.bilateralFilter(res, 15, 75, 75)

	#erosion = cv2.erode(mask, kernel, iterations=1)
	#dilation = cv2.dilate(mask, kernel2, iterations=1)

	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

	#cv2.imshow('frame', frame)

	#cv2.imshow('smoothed', smoothed)
	#cv2.imshow('gauss', gauss)
	#cv2.imshow('median', median)
	#cv2.imshow('bilateral', bilateral)

	#cv2.imshow('erosion', erosion)
	#cv2.imshow('dilation', dilation)

	cv2.imshow('mask', mask)
	cv2.imshow('opening', opening)
	cv2.imshow('closing', closing)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
cap.release()