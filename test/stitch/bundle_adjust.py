from least_squares import LSQ
import scipy.optimize as opt

class Adjuster:
	def __init__(self, images, matches, edge_array, ransac_array):
		self.images = images
		self.matches = matches
		self.lsq = LSQ(images, matches, edge_array, ransac_array)

		self.init_zoom_value = 1
		self.init_lens_value = 0

		self.lens_bounds = (0, 6)
		self.zoom_bounds = (0, None)

	def global_adjust(self):
		# since it is bounded, we can use : 'L-BFGS-B', 'TNC' and 'SLSQP'
		result = opt.minimize_scalar(self.bulk, bounds=(0, 6))
		print(result.x, result.success, result.nit)

		#self.apply()

	def bulk(self, lens_strength):
		for i in self.images:
			i.update_params(lens_strength)

		return self.lsq.total(lens_strength)

	def apply(self):
		for i in self.images:
			i.apply_changes()
			i.distort()