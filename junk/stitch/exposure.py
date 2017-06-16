import cv2, sys
import numpy as np

def gaussPyramid(image):
	gauss = image.copy()
	pyr = [gauss]

	for i in range(6):
		gauss = cv2.pyrDown(gauss)
		pyr.append(gauss)

	return pyr

def laplacePyramid(image):
	gauss = image.copy()
	gaussPyr = gaussPyramid(image)

	laplacePyr = [gaussPyr[5]]
	for i in range(5, 0, -1):
		shape = gaussPyr[i - 1].shape
		size = (shape[1], shape[0])
		gauss = cv2.pyrUp(gaussPyr[i], dstsize=size)
		laplace = cv2.subtract(gaussPyr[i - 1], gauss)
		laplacePyr.append(laplace)

	return laplacePyr

def gammaCorrect(image1, image2, mask1, mask2):
	gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

	mask = cv2.bitwise_and(mask1, mask2)

	diff = cv2.absdiff(gray1, gray2)
	diff = cv2.bitwise_and(mask, diff)

	_, diffthresh = cv2.threshold(diff, 20, 255, cv2.THRESH_TOZERO_INV)
	_, diffMask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY_INV)
	diffMask = cv2.bitwise_and(diffMask, mask)
	low, high, best = 0, 3, 1
	oldLow, oldHigh = -1, -1

	valueCut1 = cv2.bitwise_and(image1, image1, mask=diffMask)
	valueCut2 = cv2.bitwise_and(image2, image2, mask=diffMask)

	cv2.imshow("temp", diffMask)
	cv2.waitKey(0)

	while abs(oldLow - low) > 0.001 or abs(oldHigh - high) > 0.001:
		test = (low + high) / 2
		testL = (low + test) / 2
		testH = (high + test) / 2

		sumL = np.sum((valueCut1 - applyGammaCorrect(valueCut2, testL))**2)
		sumH = np.sum((valueCut1 - applyGammaCorrect(valueCut2, testH))**2)

		print(low, high)
		oldLow = low
		oldHigh = high
		if sumL < sumH:
			high = testH
			best = testL
		else:
			low = testL
			best = testH

	return applyGammaCorrect(image2, best)

def applyGammaCorrect(image, gamma):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def mergeTemp(p):
	result = None

	for i in range(6):
		merge = p[i]

		if result is None:
			result = merge
		else:
			shape = merge.shape
			size = (shape[1], shape[0])
			result = cv2.pyrUp(result, dstsize=size)
			result = cv2.add(result, merge)

	return result


def mergePyramids(p1, p2, mp1, mp2):
	result = None
	print(len(p1), len(mp1))
	for i in range(6):
		print(p1[i].shape, p2[i].shape, mp1[i].shape, mp2[i].shape)

	for i in range(6):
		cut1 = cv2.bitwise_and(p1[i], p1[i], mask=mp1[i])
		cut2 = cv2.bitwise_and(p2[i], p2[i], mask=mp2[i])
		merge = cv2.add(cut1, cut2)
		cv2.imshow("test", cut1)
		cv2.waitKey(0)
		cv2.imshow("test", cut2)
		cv2.waitKey(0)
		cv2.imshow("test", merge)
		cv2.waitKey(0)

		if result is None:
			result = merge
		else:
			shape = merge.shape
			size = (shape[1], shape[0])
			result = cv2.pyrUp(result, dstsize=size)
			result = cv2.add(result, merge)

	return result

def compensate_exposure(image1, image2, mask1, mask2):
	hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
	hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

	mask = cv2.bitwise_and(mask1, mask2)

	value1 = hsv1[:, :, 2]
	value2 = hsv2[:, :, 2]

	minValue = np.amin(value2)
	maxValue = np.amax(value2)

	diff = cv2.absdiff(value1, value2)
	diff = cv2.bitwise_and(mask, diff)

	_, diffMask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY_INV)
	diffMask = cv2.bitwise_and(diffMask, mask)

	cv2.imshow("hsv1", value1)
	cv2.waitKey(0)
	cv2.imshow("hsv2", value2)
	cv2.waitKey(0)
	cv2.imshow("diff", diff)
	cv2.waitKey(0)
	cv2.imshow("diffmask", diffMask)
	cv2.waitKey(0)

	subhsv1 = cv2.bitwise_and(value1, diffMask)
	subhsv2 = cv2.bitwise_and(value2, diffMask)

	sum1 = np.sum(subhsv1)
	shape = image1.shape

	valueArray = []

	for alpha in range(80, 120):
		temp = np.copy(hsv2).astype(float)
		temp[:,:,2] = (temp[:,:,2] * alpha / 100).astype(np.uint8)
		ret = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
		cv2.imshow("result" + str(alpha), ret)
		cv2.waitKey(0)
		tempdiff = cv2.absdiff(hsv1[:, :, 2], temp[:, :, 2])
		tempdiff = cv2.bitwise_and(mask, tempdiff)
		cv2.imshow("diff" + str(alpha), ret)
		cv2.waitKey(0)
	'''

	for i in range(256):
		bins = np.bincount(value2[value1 == i], minlength=256)

		for j in range(256):
			if bins[j] > 0:
				valueArray.append([bins[j], i, j])

	bounds = (0, 4)

	for i in range(50, 80):
		print(i / 100, difference(valueArray, i / 100, minValue, maxValue))

	alpha, beta = getAlphaBeta(valueArray, minValue, maxValue, bounds)

	hsv2[:, :, 2][mask2 == 255] *= alpha
	hsv2[:, :, 2][mask2 == 255] += beta

	return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
	'''

def getAlphaBeta(valueArray, minValue, maxValue, bounds):
	low, high = bounds
	oldAlpha, newAlpha, beta = 0, (high + low) / 2, 0

	while abs(newAlpha - oldAlpha) > 0.001:
		oldAlpha = newAlpha
		low, high, beta = findBest(valueArray, minValue, maxValue, low, high, beta)
		newAlpha = (high + low) / 2

def findBest(valueArray, minValue, maxValue, low, high, beta):
	alpha1, alpha2 = (high + 3 * low) / 4, (3 * high + low) / 4
	sum1, beta1 = difference(valueArray, alpha1, minValue, maxValue)
	sum2, beta2 = difference(valueArray, alpha2, minValue, maxValue)

	if sum1 < sum2:
		return low, (high + low) / 2, beta1
	elif sum1 > sum2:
		return (high + low) / 2, high, beta2
	else:
		return alpha1, alpha2, beta

def difference(valueArray, alpha, minValue, maxValue):
	betaMin = - int(round(alpha * minValue))
	betaMax = 255 - int(round(alpha * maxValue))

	if betaMin > betaMax:
		return None, None

	bestBeta = 0
	bestSum = sys.maxsize

	for beta in range(betaMin, betaMax + 1):
		temp = 0

		for count, v1, v2 in valueArray:
			temp += int(abs(v1 - round(alpha * v2) - beta))**2 * count

		if temp < bestSum:
			bestSum = temp
			bestBeta = beta

	return bestSum, bestBeta