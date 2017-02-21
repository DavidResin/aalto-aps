import argparse, cv2, glob, imutils, re, sys
import numpy as np

from collections import deque
from image_warp import Image_Warp
from linker import Linker
from matplotlib import pyplot as plt
from stitcher import Stitcher
from tree import TupleTree

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="Path to the image folder")
ap.add_argument("-m", "--main", required=False, help="Name of the main image (defaults to first image if not specified)")
args = vars(ap.parse_args())

filenames = glob.glob("./" + args["folder"] + "/*.jpg")

if not filenames:
	sys.exit("No images to stitch")

mainImage = None
images = []
stitcher = Stitcher()
image_count = len(filenames)

# We read the images, and put them in a table with their name and desciptors
for f in filenames:
	image = cv2.imread(f)
	small = imutils.resize(image, width=400)
	element = (f, small, stitcher.detectAndDescribe(small))
	images.append(element)

	if mainImage is None and args["main"] and re.search("[/.*|^]" + args["main"] + "$", f):
		mainImage = element

# The main image is the first one if none was specified or found
if mainImage is None:
	mainImage = images[0]

# We create a table that we fill with the keypoint matches
ransacArray = np.empty((image_count, image_count), dtype=object)

for i in range(image_count):
	for j in range(image_count):
		if i == j:
			temp = (None, None, [])
		else:
			temp = stitcher.ransac(images[j][2], images[i][2])
		ransacArray[i][j] = temp

# We find the best edges and build a tree using the results
linker = Linker()
edge_array = linker.solve(ransacArray, 2)
tree = linker.edgesToTree(firstelem=mainImage, firstdata=np.identity(3), array=images, edge_table=edge_array, data_table=ransacArray)
data_set = tree.flatten()
reduced_set = []
iw = Image_Warp()

for elem in data_set:
	transformed = iw.homography_warp(elem[1], elem[4])
	reduced_set.append(transformed)

positioned_images = iw.position_images(reduced_set)

final_images = []
biggest = [0, 0]

for p in positioned_images:
	(img, trans) = p
	final_images.append(iw.apply_translation(img, trans))

	for i in range(2):
		if img.shape[:2][i] > biggest[i]:
			biggest[i] = img.shape[:2][i]

result = final_images[0]

for i in final_images[1:]:
	result = iw.copy_over(result, i, tuple(biggest))

cv2.imwrite("test.jpg", result)
