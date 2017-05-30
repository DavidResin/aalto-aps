import image_warp, math
import numpy as np
import distortion as dist

class LSQ:
	def __init__(self, images, matches):
		self.images = images
		self.matches = matches

	def values(self):
		return [self.distance(i) for i in range(len(self.matches))]

	def distance(self, matchIndex):
		imIdx1, imIdx2, featIdx1, featIdx2 = self.matches[matchIndex]
		feat1 = self.moveFeature(imIdx1, featIdx1)
		feat2 = self.moveFeature(imIdx2, featIdx2)
		return math.sqrt((feat1[0] - feat2[0])**2 + (feat1[1] - feat2[1])**2)

	def moveFeature(self, imIdx, featIdx):
		image = self.images[imIdx]
		x, y = image.descriptor[0][featIdx]
		x, y = dist.lensCorrectParams(x, y, image.temp_params)
		x = x - image.temp_params.lens_offset[0]
		y = y - image.temp_params.lens_offset[1]
		vector = np.array([x, y, 1])
		return image_warp.from_homogeneous(np.atleast_2d(image.matrix.dot(np.transpose(vector))).T)

	def total(self):
		self.values = self.values()
		return sum(self.values)