import copy, image_warp, math, stitcher
import numpy as np
import distortion as dist

class LSQ:
	def __init__(self, images, matches, edge_array, ransac_array):
		self.images = images
		self.matches = matches
		self.edge_array = edge_array
		self.ransac_array = ransac_array

	def update_descriptor(self, image, strength):
		image.update_params(strength)
		new_descriptor = copy.deepcopy(image.descriptor)
		keypoints = []
		k, f = new_descriptor

		for x, y in k:
			keypoints.append(dist.lensCorrectParams(x, y, image.temp_params))

		image.lens_descriptor = keypoints, f

	def get_row_values(self, i):
		total = 0

		for j in range(len(self.edge_array)):
			if self.edge_array[i, j]:
				matches, matrix, status = self.ransac_array[j, i]
				newMatrix, newStatus = stitcher.getHomography(self.images[i].lens_descriptor[0], self.images[j].lens_descriptor[0], matches)
				self.ransac_array[j, i] = matches, newMatrix, newStatus
				total += newMatrix[2][0]**2 + newMatrix[2][1]**2 + newMatrix[0][1]**2 + newMatrix[1][0]**2

		return total

	def global_total(self, strength):
		for image in self.images:
			self.update_descriptor(image, strength)

		total = 0

		for i in range(len(self.edge_array)):
			total += self.get_row_values(i)

		return total

	def single_total(self, strength, image):
		self.update_descriptor(image, strength)

		return self.get_row_values(image.index)

	#DEPRECATED STUFF
		
	def values(self):
		return [self.distance(i) for i in range(len(self.matches))]

	#MOVE ALL FEATURES BEFORE
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
	'''
	def total(self):
		return sum(self.values())
	'''