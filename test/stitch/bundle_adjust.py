import copy, image_warp, math, stitcher
import distortion as dist
import numpy as np
import scipy.optimize as opt

# This class takes care of bundle adjustment
class Adjuster:
	# Initialises the adjuster
	def __init__(self, images, edge_array, ransac_array, details):
		self.images = images
		self.edge_array = edge_array
		self.ransac_array = ransac_array
		self.lens_bounds = (0, 6)
		self.details = details

	# Refines the lens distortion on all images
	def global_adjust(self):
		opt.minimize_scalar(self.global_total, bounds=self.lens_bounds)
		self.apply()

	# Finds the best lens distortion for each image
	def individual_adjust(self):
		for a in range(3):
			for i in self.images:
				print(a, i.index)
				opt.minimize_scalar(self.single_total, args=i, bounds=self.lens_bounds)

		self.apply()

	# Applies the final lens distortion to every image
	def apply(self):
		for i in self.images:
			i.apply_changes()
			i.distort(self.details)

	# Sets the affine transformation for each image
	def set_transforms(self):
		for i in range(len(self.edge_array)):
			for j in range(len(self.edge_array)):
				if self.edge_array[i, j]:
					matches, matrix, status = self.ransac_array[j, i]
					newMatrix = stitcher.getAffine(self.images[i].lens_descriptor[0], self.images[j].lens_descriptor[0], matches)
					self.ransac_array[j, i] = matches, newMatrix, status

	# Updates the coordinates of the features of the image according to its lens distortion strength
	def update_descriptor(self, image, strength):
		image.update_params(strength)
		new_descriptor = copy.deepcopy(image.descriptor)
		keypoints = []
		k, f = new_descriptor

		for x, y in k:
			keypoints.append(dist.lensCorrectParams(x, y, image.temp_params))

		image.lens_descriptor = keypoints, f

	# Gets all the squared flip and shear values (values [0, 1], [2, 0] and [2, 1] in the matrix)
	def get_row_values(self, i):
		total = 0

		for j in range(len(self.edge_array)):
			if self.edge_array[i, j]:
				matches, matrix, status = self.ransac_array[j, i]
				newMatrix, newStatus = stitcher.getHomography(self.images[i].lens_descriptor[0], self.images[j].lens_descriptor[0], matches)
				self.ransac_array[j, i] = matches, newMatrix, newStatus
				total += newMatrix[2][0]**2 + newMatrix[2][1]**2 + newMatrix[0][1]**2 + newMatrix[1][0]**2

		return total

	# Updates all descriptors and gets all squared matrix values we're interested to minimize
	def global_total(self, strength):
		for image in self.images:
			self.update_descriptor(image, strength)

		total = 0

		for i in range(len(self.edge_array)):
			total += self.get_row_values(i)

		return total

	# Updates the descriptor of the given image and gets the squared matrix values of its transformations with its neighbors
	def single_total(self, strength, image):
		self.update_descriptor(image, strength)

		return self.get_row_values(image.index)