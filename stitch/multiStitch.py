import argparse, cv2, glob, imutils, linker, os, re, stitcher, sys, timeit
import image_warp as iw
import numpy as np

from bundle_adjust import Adjuster
from collections import deque
from image_data import Image_Data
from matplotlib import pyplot as plt
from timeit import default_timer as timer

# Sets up the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="inputs/", help="Path for inputs, default is \'inputs/\'")
ap.add_argument("-o", "--output", default="outputs/", help="Path for outputs, default is \'outputs/\'")
ap.add_argument("-m", "--main", required=False, help="Name of the main image (defaults to the middle image if not specified)")
ap.add_argument("-s", "--spherical", action='store_true', default=False, help="Generates a spherical projection instead of the standard projection")
ap.add_argument("-d", "--details", action='store_true', default=False, help="Outputs intermediate data like masks, transformed and lens-warped images, as well as working data from the seam finder")
ap.add_argument("-z", "--zoom", default=0.5, help="Size factor for the input images, the program can take a long time and even fail if the value is too big. Default is 0.5")
args = vars(ap.parse_args())

if not os.path.isdir(args["output"]):
	os.mkdir(args["output"])

time_start = timer()
print("Setting up images....")

# Deletes everything in the output folder
to_flush = glob.glob("./" + args["output"] + "/*")

for f in to_flush:
	os.remove(f)

# Gets the names of the input images, exits the program if none are found
filenames = glob.glob("./" + args["input"] + "/*.jpg")

if not filenames:
	sys.exit("No images found")

spherical_proj = args["spherical"]
print_details = args["details"]

mainImage = None
images = []
image_count = len(filenames)

# We read the images, and set up image data elements that we put in a table
for i in range(image_count):
	f = filenames[i]
	element = Image_Data(i, f, args["zoom"])
	images.append(element)
	print("Found", f)

	if mainImage is None and args["main"] and re.search("[/.*|^]" + args["main"] + "$", f):
		mainImage = element

os.chdir(args["output"])

# The main image is the first one if none was specified or found
if mainImage is None:
	mainImage = images[int(len(images) / 2)]

# The initial transformation is the identity of course
mainImage.matrix = np.identity(3)

time_images = timer()
print("Image setup complete:", str(time_images - time_start), "s")
print("Extracting features....")

# We create a table that we fill with the keypoint matches
ransacArray = np.empty((image_count, image_count), dtype=object)

for i in range(image_count):
	ransacArray[i][i] = (None, None, [])

	for j in range(i + 1, image_count):
		ransacArray[i][j] = stitcher.ransac(images[j].descriptor, images[i].descriptor)
		ransacArray[j][i] = stitcher.ransac(images[i].descriptor, images[j].descriptor)

# We find the best edges and build a tree using the results
edge_array = linker.solve(ransacArray, 2)

time_features = timer()
print("Feature extraction complete:", str(time_features - time_images), "s")

# We apply bundle adjustment if spherical projection is chosen
if args["spherical"]:
	print("Bundle adjustment....")

	try:
		adjuster = Adjuster(images, edge_array, ransacArray, args["details"])
		adjuster.global_adjust()
		#adjuster.individual_adjust()
		adjuster.set_transforms()

		time_bundle = timer()
		time_features = time_bundle
	except:
		sys.exit("Transformation error. Try to reduce the amount of images and/or the zoom factor.")
		
	print("Bundle adjustment complete:", str(time_bundle - time_features), "s")
else:
	for i in images:
		i.update_params(0)
		i.apply_changes()
		i.distort(args["details"])

print("Getting final transformations....")

# We reduce the tree to spread the transformations down the tree
tree = linker.edgesToTree(firstelem=mainImage, array=images, edge_table=edge_array, data_table=ransacArray)

time_tree = timer()
print("Final transformations computed:", str(time_tree - time_features), "s")
print("Warping images....")

# Apply the homographies and repositioning and combine all images
try:
	iw.homography_warp(images)
	iw.position_images(images)
	iw.apply_translation(images, args["details"])
	result = iw.copy_over(images, args["details"])
except:
	sys.exit("Transformation error. Try to reduce the amount of images and/or the zoom factor.")

time_warp = timer()
print("Images warped:", str(time_warp - time_tree), "s")
print("TOTAL RUNTIME:", str(time_warp - time_start), "s")

# We print the final result
cv2.imwrite("result.jpg", result)