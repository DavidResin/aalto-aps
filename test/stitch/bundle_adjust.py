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