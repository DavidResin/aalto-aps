import copy, cv2, imutils, stitcher
import numpy as np
import distortion as dist

class Image_Params:
	def __init__(self):
		self.lens_offset = (0, 0)
		self.lens_strength = 0
		self.zoom_strength = 1
		self.theta_inv = 0
		self.correction_radius = 0
		self.image_size = None

		self.newW = 0
		self.newH = 0

		self.lens_zoomX = 1
		self.lens_zoomY = 1

	def update(self, image_size, lens_value=0, zoom_value=1):
		h, w = image_size
		padX, padY, theta_inv, cR = dist.paddings(w, h, lens_value)

		self.image_size = image_size
		self.lens_offset = (padX, padY)
		self.lens_strength = lens_value
		self.zoom_strength = zoom_value
		self.theta_inv = theta_inv
		self.correction_radius = cR

		self.newW = w + 2 * padX
		self.newH = h + 2 * padY

		self.lens_zoomX = image_size[1] / self.newW
		self.lens_zoomY = image_size[0] / self.newH

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
		self.descriptor = stitcher.detectAndDescribe(self.image_resized)
		self.correspondances = [[] for i in range(len(self.descriptor[0]))]
		self.matrix = None
		self.image_transformed = None
		self.offset = None
		self.new_size = None

		self.params = Image_Params()
		self.temp_params = Image_Params()

	def apply_changes(self):
		self.params = copy.deepcopy(self.temp_params)

	def reset_temp_changes(self):
		self.temp_params = copy.deepcopy(self.params)

	def update_params(self, lens_value=0, zoom_value=1):
		self.temp_params.update(self.image_size, lens_value, zoom_value)

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