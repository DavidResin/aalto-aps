import argparse, cv2, glob, imutils, linker, re, stitcher, sys
import image_warp as iw
import numpy as np

from collections import deque
from bundle_adjust import Adjuster
from image_data import Image_Data
from least_squares import LSQ
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="Path to the image folder")
ap.add_argument("-m", "--main", required=False, help="Name of the main image (defaults to first image if not specified)")
args = vars(ap.parse_args())

filenames = glob.glob("./" + args["folder"] + "/*.jpg")

if not filenames:
	sys.exit("No images to stitch")

mainImage = None
images = []
image_count = len(filenames)

# We read the images, and put them in a table with their name and desciptors
for i in range(len(filenames)):
	f = filenames[i]
	element = Image_Data(i, f, 0.5)
	#image = cv2.imread(f)
	#small = imutils.resize(image, width=400)
	#element = (f, image, stitcher.detectAndDescribe(image))
	images.append(element)

	if mainImage is None and args["main"] and re.search("[/.*|^]" + args["main"] + "$", f):
		mainImage = element

# The main image is the first one if none was specified or found
if mainImage is None:
	mainImage = images[0]

mainImage.matrix = np.identity(3)

# We create a table that we fill with the keypoint matches
ransacArray = np.empty((image_count, image_count), dtype=object)
matchArray = []

for i in range(image_count):
	ransacArray[i][i] = (None, None, [])

	for j in range(i + 1, image_count):
		ransacArray[i][j] = stitcher.ransac(images[j].descriptor, images[i].descriptor)
		ransacArray[j][i] = stitcher.ransac(images[i].descriptor, images[j].descriptor)
		images[i].cross(images[j])
		images[j].cross(images[i])

for i in range(image_count):
	corr = images[i].correspondances

	for j in range(len(corr)):
		c = corr[j]

		for (img_idx, feat_idx) in c:
	
			imgIdx1 = i
			imgIdx2 = img_idx
			featIdx1 = j
			featIdx2 = feat_idx

			if imgIdx1 > imgIdx2:
				imgIdx1, imgIdx2 = imgIdx2, imgIdx1
				featIdx1, featIdx2 = featIdx2, featIdx1

			match = (imgIdx1, imgIdx2, featIdx1, featIdx2)

			if match not in matchArray:
				matchArray.append(match)

# We find the best edges and build a tree using the results
edge_array = linker.solve(ransacArray, 2)
tree = linker.edgesToTree(firstelem=mainImage, array=images, edge_table=edge_array, data_table=ransacArray)
images = tree.flatten()

adjuster = Adjuster(images, matchArray)
adjuster.global_adjust()

iw.homography_warp(images)
iw.position_images(images)
iw.apply_translation(images)
result = iw.copy_over(images[::-1])
cv2.imwrite("test.jpg", result)