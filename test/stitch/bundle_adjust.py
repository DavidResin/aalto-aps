from least_squares import LSQ
import scipy.optimize as opt
import stitcher

class Adjuster:
	def __init__(self, images, matches, edge_array, ransac_array):
		self.images = images
		self.matches = matches
		self.edge_array = edge_array
		self.ransac_array = ransac_array
		self.lsq = LSQ(images, matches, edge_array, ransac_array)
		self.lens_bounds = (0, 6)

	def global_adjust(self):
		opt.minimize_scalar(self.lsq.global_total, bounds=self.lens_bounds)
		self.apply()

	def individual_adjust(self):
		for a in range(5):
			for i in self.images:
				print(a, i.index)
				opt.minimize_scalar(self.lsq.single_total, args=i, bounds=self.lens_bounds)

		self.apply()

	def set_transforms(self):
		for i in range(len(self.edge_array)):
			for j in range(len(self.edge_array)):
				if self.edge_array[i, j]:
					matches, matrix, status = self.ransac_array[j, i]
					newMatrix = stitcher.getAffine(self.images[i].lens_descriptor[0], self.images[j].lens_descriptor[0], matches)
					self.ransac_array[j, i] = matches, newMatrix, status

	def apply(self):
		for i in self.images:
			i.apply_changes()
			i.distort()