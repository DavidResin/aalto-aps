import cv2, watershed
import exposure as ex
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
		src = image_data.lens_image
		mask = image_data.lens_mask
		matrix = image_data.matrix
		h, w = src.shape[:2]
		corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
		center = np.array([[w / 2, h / 2]]) 
		projected_corners = transform_via_homography(corners, matrix)
		bb = bounding_box(projected_corners)
		put_in_corner = np.array([[1, 0, -bb[0]], [0, 1, -bb[1]], [0, 0, 1]])
		dimensions = (bb[2] - bb[0], bb[3] - bb[1])
		new_matrix = np.float32(np.dot(put_in_corner, matrix))

		image_data.image_transformed = cv2.warpPerspective(src, new_matrix, dimensions, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
		image_data.mask_transformed = cv2.warpPerspective(mask, new_matrix, dimensions, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
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
		img = np.zeros(image_data.new_size + (image_data.channels,), np.uint8)
		mask = np.zeros(image_data.new_size, np.uint8)
		img[tY:tY + h, tX:tX + w] = image_data.image_transformed
		mask[tY:tY + h, tX:tX + w] = image_data.mask_transformed
		image_data.image_transformed = img
		image_data.mask_transformed = mask

		cv2.imwrite("test" +str(image_data.index)+".jpg", image_data.image_transformed)
		cv2.imwrite("mask" +str(image_data.index)+".jpg", image_data.mask_transformed)

def copy_over(images):
	image1 = images[0].image_transformed
	mask1 = images[0].mask_transformed

	for image_data in images[1:]:
		mask2 = image_data.mask_transformed
		image2 = image_data.image_transformed

		own_mask1 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
		own_mask2 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))

		zone1, zone2 = watershed.cut(image1, image2, mask1, mask2)

		sharp_mask1 = cv2.bitwise_or(own_mask1, zone1)
		sharp_mask2 = cv2.bitwise_or(own_mask2, zone2)

		mask = cv2.bitwise_or(mask1, mask2)

		temp1 = np.copy(mask1)
		innerFade1 = np.copy(mask1)
		temp2 = np.copy(mask2)
		innerFade2 = np.copy(mask2)


		for i in range(1, 32):
			line1 = watershed.innerEdge(temp1)
			line2 = watershed.innerEdge(temp2)
			temp1 -= line1
			temp2 -= line2
			line1[line1 == 255] -= i * 8
			line2[line2 == 255] -= i * 8
			innerFade1 -= line1
			innerFade2 -= line2

		tempMask1 = np.copy(sharp_mask1)
		tempMask2 = np.copy(sharp_mask2)
		fill1 = np.copy(sharp_mask1)
		fill2 = np.copy(sharp_mask2)
		temp1 = np.copy(fill1)
		temp2 = np.copy(fill2)

		for i in range(1, 16):
			line1 = watershed.outerEdge(fill1)
			line2 = watershed.outerEdge(fill2)
			fill1 = cv2.bitwise_or(fill1, line1)
			fill2 = cv2.bitwise_or(fill2, line2)
			line1[line1 == 255] -= i * 16
			line2[line2 == 255] -= i * 16
			tempMask1 = cv2.bitwise_or(tempMask1, line1)
			tempMask2 = cv2.bitwise_or(tempMask2, line2)


		innerFade1 = innerFade1.astype(float) / 255
		innerFade2 = innerFade2.astype(float) / 255


		tempMask1 = cv2.filter2D(tempMask1, -1, np.array([[0.5]]))
		tempMask2 = cv2.filter2D(tempMask2, -1, np.array([[0.5]]))

		tempMask1 = cv2.bitwise_and(sharp_mask2, tempMask1)
		tempMask2 = cv2.bitwise_and(sharp_mask1, tempMask2)
		tempMask1 = cv2.multiply(innerFade1, tempMask1.astype(float)).astype(np.uint8)
		tempMask2 = cv2.multiply(innerFade2, tempMask2.astype(float)).astype(np.uint8)

		sharp_mask1 = cv2.add(tempMask1, sharp_mask1)
		sharp_mask2 = cv2.add(tempMask2, sharp_mask2)
		sharp_mask1 = cv2.subtract(sharp_mask1, tempMask2)
		sharp_mask2 = cv2.subtract(sharp_mask2, tempMask1)

		crossMask1 = cv2.bitwise_and(fill1, cv2.bitwise_not(temp1))
		crossMask2 = cv2.bitwise_and(fill2, cv2.bitwise_not(temp2))
		crossMask = cv2.bitwise_or(crossMask1, crossMask2)
		crossMask = cv2.bitwise_and(crossMask, mask)

		alpha1 = sharp_mask1.astype(float) / 255
		alpha2 = sharp_mask2.astype(float) / 255

		alpha1 = cv2.merge((alpha1, alpha1, alpha1))
		alpha2 = cv2.merge((alpha2, alpha2, alpha2))

		final_image1 = cv2.bitwise_and(image1, image1, mask=sharp_mask1).astype(float)
		final_image2 = cv2.bitwise_and(image2, image2, mask=sharp_mask2).astype(float)

		final_image1 = cv2.multiply(alpha1, final_image1).astype(np.uint8)
		final_image2 = cv2.multiply(alpha2, final_image2).astype(np.uint8)
		
		image1 = cv2.add(final_image1, final_image2)
		mask1 = cv2.add(sharp_mask1, sharp_mask2)

	return image1