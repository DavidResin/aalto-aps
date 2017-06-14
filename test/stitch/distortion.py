import math

def lensCorrect(x, y, halfW, halfH, cR, theta_inv):
	cX = x - halfW
	cY = y - halfH

	theta = factor(cX, cY, cR)

	newX = halfW + theta * cX * theta_inv
	newY = halfH + theta * cY * theta_inv

	return (int(newX), int(newY))

def lensCorrectParams(x, y, params):
	halfW = params.image_size[1] / 2
	halfH = params.image_size[0] / 2
	cR = params.correction_radius
	theta_inv = params.theta_inv
	xN, yN = lensCorrect(x, y, halfW, halfH, cR, theta_inv)
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
	padX, _ = lensCorrect(0, h / 2, w / 2, h / 2, cR, theta_inv)
	_, padY = lensCorrect(w / 2, 0, w / 2, h / 2, cR, theta_inv)

	return (-int(padX), -int(padY), theta_inv, cR)

def kCorrect(x, y, halfW, halfH, f, ks):
	r2 = (x - halfW)**2 + (y - halfH)**2
	coeff = 1

	for i in range(len(ks)):
		coeff += ks[i] * r2**(i + 1)

	coeff *= f

	xF = x * coeff
	yF = y * coeff

	return int(xF), int(yF)