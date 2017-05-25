import cv2, imutils, stitcher
import numpy as np

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
		self.descriptor = stitcher.detectAndDescribe(self.image_resized)
		self.correspondances = [[] for i in range(len(self.descriptor[0]))]
		self.matrix = None
		self.image_transformed = None
		self.offset = None
		self.new_size = None

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