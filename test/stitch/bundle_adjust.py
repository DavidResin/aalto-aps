from least_squares import LSQ
import scipy.optimize as opt

class Adjuster:
	def __init__(self, images, matches):
		self.images = images
		self.matches = matches
		self.lsq = LSQ(images, matches)

		self.init_zoom_value = 1
		self.init_lens_value = 0
		self.zoom_value = self.init_zoom_value
		self.lens_value = self.init_lens_value
		self.counter_init = 5
		self.lens_limit = 6

	def global_adjust(self):
		switch = True
		zoom_contrib = True
		lens_contrib = True

		while zoom_contrib or lens_contrib:
			temp = opt.minimize(self.bulk_lens, self.lens_value, method='Nelder-Mead')

			if temp == self.lens_value:
				lens_contrib = False
			else:
				lens_contrib = True
				self.lens_value = temp

			temp = opt.minimize(self.bulk_zoom, self.zoom_value, method='Nelder-Mead')

			if temp == self.zoom_value:
				zoom_contrib = False
			else:
				zoom_contrib = True
				self.zoom_value = temp

	def bulk_lens(self, value):
		for i in self.images:
			i.update_params(value, self.zoom_value)

		return self.lsq.total()

	def bulk_zoom(self, value):
		for i in self.images:
			i.update_params(self.lens_value, value)

		return self.lsq.total()
