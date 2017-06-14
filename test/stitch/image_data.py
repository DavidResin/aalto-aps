import copy, cv2, imutils, math, stitcher
import numpy as np
import distortion as dist

class Image_Params:
	def __init__(self, image_size):
		self.lens_offset = (0, 0)
		self.lens_strength = 0
		self.theta_inv = 0
		self.correction_radius = 0
		self.image_size = image_size

		self.newW = 0
		self.newH = 0

	def update(self, lens_value):
		h, w = self.image_size
		padX, padY, theta_inv, cR = dist.paddings(w, h, lens_value)
		
		self.lens_offset = (padX, padY)
		self.lens_strength = lens_value
		self.theta_inv = theta_inv
		self.correction_radius = cR

		self.newW = w + 2 * padX
		self.newH = h + 2 * padY

class Image_Data():
	def __init__(self, index, filename, ratio=1):
		self.index = index
		self.filename = filename
		self.image_orig = cv2.imread(filename)

		if ratio <= 0:
			ratio = 1

		self.ratio = ratio
		_, w = self.image_orig.shape[:2]

		self.image_resized = imutils.resize(self.image_orig, width=int(w * self.ratio))
		self.image_size = self.image_resized.shape[:2]
		self.channels = self.image_resized.shape[2]
		self.mask_resized = np.zeros(self.image_size, dtype=np.uint8).fill(255)
		self.descriptor = stitcher.detectAndDescribe(self.image_resized)
		self.lens_descriptor = None
		self.correspondances = [[] for i in range(len(self.descriptor[0]))]
		self.lens_image = None
		self.lens_mask = None
		self.matrix = None
		self.image_transformed = None
		self.mask_transformed = None
		self.offset = None
		self.new_size = None
		self.center = None
		
		self.params = Image_Params(self.image_size)
		self.temp_params = Image_Params(self.image_size)

	def apply_changes(self):
		self.params = copy.deepcopy(self.temp_params)

	def distort(self):
		params = self.params
		padX, padY = params.lens_offset

		newHalfW = params.newW / 2
		newHalfH = params.newH / 2

		image = cv2.resize(self.image_resized, (params.newW, params.newH))
		newImage = np.zeros((params.newH, params.newW, self.channels), np.uint8)
		newMask = np.zeros((params.newH, params.newW, 1), np.uint8)

		if params.lens_strength > 0:
			log = math.ceil(math.log2(params.lens_strength + 2))
		else:
			log = 1

		for x in range(params.newW):
			for y in range(params.newH):
				newX, newY = dist.lensCorrectParams(x, y, params)
				tempX = padX + newX
				tempY = padY + newY
				
				for i in range(log):
					for j in range(log):
						if tempX + i < params.newW and tempY + j < params.newH:
							newImage[tempY + j][tempX + i] = image[y][x]
							newMask[tempY + j][tempX + i] = 255
	
		self.lens_image = imutils.resize(newImage, width=int(params.newW * params.zoom_strength))
		self.lens_mask = imutils.resize(newMask, width=int(params.newW * params.zoom_strength))

	def reset_temp_changes(self):
		self.temp_params = copy.deepcopy(self.params)

	def update_params(self, lens_strength):
		self.temp_params.update(lens_strength)

	def cross(self, other, ratio=0.75, reprojThresh=4.0):
		kp1, feat1 = self.descriptor
		kp2, feat2 = other.descriptor

		crossdata = stitcher.matchKeypoints(kp1, kp2, feat1, feat2, ratio, reprojThresh)

		if crossdata is not None and crossdata[2] is not None:
			(matches, matrix, status) = crossdata

			for ((m1, m2), s) in zip(matches, status):
				if s == 1:
					self.correspondances[m2].append((other.index, m1))

		return crossdata