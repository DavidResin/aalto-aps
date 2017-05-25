import cv2
import numpy as np

# Convert a vector of non-homogeneous 2D points to a vector of homogenehous 2D points.
def to_homogeneous(non_homo):
	shape = non_homo.shape
	return np.c_[non_homo, np.ones(shape[0])]

# Convert a vector of homogeneous 2D points to a vector of non-homogenehous 2D points.
def from_homogeneous(homo):
	temp = homo.T
	for t in temp:
		t /= float(t[2])
	return np.delete(temp.T, (2), axis=0)

# Transform a vector of 2D non-homogeneous points via an homography.
def transform_via_homography(points, H):
	homo = to_homogeneous(points)
	return from_homogeneous(H.dot(homo.T)).astype(int)

# Find the bounding box of an array of 2D non-homogeneous points and return (minX, minY, maxX, maxY)
def bounding_box(points):
	mins = points.min(axis=1)
	maxs = points.max(axis=1)
	return (mins[0], mins[1], maxs[0], maxs[1])

# Returns the image src warped through the homography H and the translation values to add to the image. The resulting dst image contains the entire warped image.
def homography_warp(images):
	for image_data in images:
		src = image_data.image_resized
		matrix = image_data.matrix
		h, w = src.shape[:2]
		corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
		projected = transform_via_homography(corners, matrix)
		bb = bounding_box(projected)
		put_in_corner = np.array([[1, 0, -bb[0]], [0, 1, -bb[1]], [0, 0, 1]])
		dimensions = (bb[2] - bb[0], bb[3] - bb[1])

		image_data.image_transformed = cv2.warpPerspective(src, np.float32(np.dot(put_in_corner, matrix)), dimensions)
		image_data.offset = (bb[0], bb[1])

def position_images(images):
	offset_x, offset_y, total_h, total_w = 0, 0, 0, 0

	for image_data in images:
		h, w = image_data.image_transformed.shape[:2]
		x, y = image_data.offset

		if w + x > total_w:
			total_w = w + x

		if h + y > total_h:
			total_h = h + y

		if x < offset_x:
			offset_x = x

		if y < offset_y:
			offset_y = y

	offset_x = -offset_x
	offset_y = -offset_y
	total_w += offset_x
	total_h += offset_y

	for image_data in images:
		image_data.new_size = (total_h, total_w)
		x, y = image_data.offset
		image_data.offset = (x + offset_x, y + offset_y)

def apply_translation(images):
	for image_data in images:
		tX, tY = image_data.offset
		h, w = image_data.image_transformed.shape[:2]
		img = np.zeros(image_data.new_size + (3,), np.uint8)
		img[tY:tY + h, tX:tX + w] = image_data.image_transformed
		image_data.image_transformed = img
		cv2.imwrite("test" +str(image_data.index)+".jpg", img)

def copy_over(images):
	img = images[0].image_transformed

	for image_data in images[1:]:
		src = image_data.image_transformed

		src_grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		_, mask = cv2.threshold(src_grey, 1, 255, cv2.THRESH_BINARY)

		src = cv2.bitwise_and(src, src, mask=mask)
		img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

		img = cv2.add(src, img)

	return img
	'''
	matrix = np.float32(np.array([[1, 0, 0], [0, 1, 0]]))

	src_big = cv2.warpAffine(src, matrix, shape)
	dst_big = cv2.warpAffine(dst, matrix, shape)

	src_grey = cv2.cvtColor(src_big, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(src_grey, 1, 255, cv2.THRESH_BINARY)

	new_src = cv2.bitwise_and(src_big, src_big, mask=mask)
	new_dst = cv2.bitwise_and(dst_big, dst_big, mask=cv2.bitwise_not(mask))
	return cv2.add(new_src, new_dst)
	'''