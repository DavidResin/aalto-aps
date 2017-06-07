from least_squares import LSQ
import scipy.optimize as opt

class Adjuster:
	def __init__(self, images, matches):
		self.images = images
		self.matches = matches
		self.lsq = LSQ(images, matches)

		self.init_zoom_value = 1
		self.init_lens_value = 0

		self.lens_bounds = (0, 6)
		self.zoom_bounds = (0, None)

	def global_adjust(self):
		# since it is bounded, we can use : 'L-BFGS-B', 'TNC' and 'SLSQP'
		result = opt.minimize(self.bulk, (self.init_lens_value, self.init_zoom_value), method='TNC', bounds=((0, 6), (0, None)))
		print(result.x, result.success, result.nit)

		self.lens_value, self.zoom_value = result.x
		self.bulk((0, 1))
		self.apply()

	def bulk(self, data):
		#print("bulk", data)
		lens_value, zoom_value = data
		for i in self.images:
			i.update_params(lens_value, zoom_value)

		return self.lsq.total()

	def apply(self):
		for i in self.images:
			i.apply_changes()
			i.distort()