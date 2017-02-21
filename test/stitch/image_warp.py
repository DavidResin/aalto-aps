import cv2
import numpy as np

class Image_Warp():
	def __init__(self):
		pass

	# Convert a vector of non-homogeneous 2D points to a vector of homogenehous 2D points.
	def to_homogeneous(self, non_homo):
		shape = non_homo.shape
		return np.c_[non_homo, np.ones(shape[0])]

	# Convert a vector of homogeneous 2D points to a vector of non-homogenehous 2D points.
	def from_homogeneous(self, homo):
		temp = homo.T
		for t in temp:
			t /= float(t[2])
		return np.delete(temp.T, (2), axis=0)

	# Transform a vector of 2D non-homogeneous points via an homography.
	def transform_via_homography(self, points, H):
		homo = self.to_homogeneous(points)
		return self.from_homogeneous(H.dot(homo.T)).astype(int)

	# Find the bounding box of an array of 2D non-homogeneous points and return (minX, minY, maxX, maxY)
	def bounding_box(self, points):
		mins = points.min(axis=1)
		maxs = points.max(axis=1)
		return (mins[0], mins[1], maxs[0], maxs[1])

	# Returns the image src warped through the homography H and the translation values to add to the image. The resulting dst image contains the entire warped image.
	def homography_warp(self, src, H):
		h, w = src.shape[:2]
		corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
		projected = self.transform_via_homography(corners, H)
		bb = self.bounding_box(projected)
		put_in_corner = np.array([[1, 0, -bb[0]], [0, 1, -bb[1]], [0, 0, 1]])
		dimensions = (bb[2] - bb[0], bb[3] - bb[1])
		return (cv2.warpPerspective(src, np.float32(np.dot(put_in_corner, H)), dimensions), (-bb[0], -bb[1]))

	def position_images(self, data_set):
		from_corner = []
		extremas = [0, 0]
		length = 0
		new_data = []

		for (image_data, trans) in data_set:
			from_corner.append(trans)
			length += 1

			for i in range(2):
				if trans[i] > extremas[i]:
					extremas[i] = trans[i]

		for i in range(length):
			tempX = int(extremas[0] - from_corner[i][0])
			tempY = int(extremas[1] - from_corner[i][1])
			print(tempX, tempY)

			new_data.append((data_set[i][0], (tempX, tempY)))

		return new_data

	def apply_translation(self, src, trans):
		(tX, tY) = trans
		matrix = np.float32(np.array([[1, 0, tX], [0, 1, tY]]))
		h, w = src.shape[:2]
		img = np.zeros((h + tX, w + tY, 3), np.uint8)
		img[:h, :w] = src
		img = cv2.warpAffine(img, matrix, img.shape[:2])
		return img

	def copy_over(self, src, dst, shape):
		matrix = np.float32(np.array([[1, 0, 0], [0, 1, 0]]))

		src_big = cv2.warpAffine(src, matrix, shape)
		dst_big = cv2.warpAffine(dst, matrix, shape)

		src_grey = cv2.cvtColor(src_big, cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(src_grey, 10, 255, cv2.THRESH_BINARY)

		new_src = cv2.bitwise_and(src_big, src_big, mask=mask)
		new_dst = cv2.bitwise_and(dst_big, dst_big, mask=cv2.bitwise_not(mask))
		return cv2.add(new_src, new_dst)