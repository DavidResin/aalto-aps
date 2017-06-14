import math

def lensCorrect(x, y, halfW, halfH, cR, theta_inv, zoomX, zoomY):
	cX = x - halfW
	cY = y - halfH

	theta = factor(int(cX * zoomX), int(cY * zoomY), cR)

	newX = int(halfW + theta * cX * theta_inv) * zoomX
	newY = int(halfH + theta * cY * theta_inv) * zoomY

	return (int(newX), int(newY))

def lensCorrectParams(x, y, params):
	halfW = params.image_size[1] / 2
	halfH = params.image_size[0] / 2
	cR = params.correction_radius
	theta_inv = params.theta_inv
	xN, yN = lensCorrect(x, y, halfW, halfH, cR, theta_inv, params.zoomX, params.zoomY)
	return xN + params.lens_offset[0], yN + params.lens_offset[1]

def factor(x, y, cR):
	ratio = math.sqrt(x**2 + y**2) * cR

	if ratio == 0:
		theta = 1;
	else:
		theta = math.atan(ratio) / ratio

	return theta

def paddings(w, h, strength):
	cR = strength / math.sqrt(w**2 + h**2)
	theta_inv = 1 / factor(w / 2, h / 2, cR)
	padX, _ = lensCorrect(0, h / 2, w / 2, h / 2, cR, theta_inv, 1, 1)
	_, padY = lensCorrect(w / 2, 0, w / 2, h / 2, cR, theta_inv, 1, 1)

	return (-int(padX), -int(padY), theta_inv, cR)