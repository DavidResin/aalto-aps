from stitcher import Stitcher
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="path to the first image")
ap.add_argument("-s", "--second", required=True, help="path to the second image")
args = vars(ap.parse_args())

img1 = cv2.imread(args["first"])
img2 = cv2.imread(args["second"])

img1 = imutils.resize(img1, width=400)
img2 = imutils.resize(img2, width=400)

stitcher = Stitcher()
(result, vis) = stitcher.stitch([img1, img2], showMatches=True)

cv2.imwrite("resStitch.jpg", result)

plt.subplot(2, 2, 1),plt.imshow(img1)
plt.title('Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2),plt.imshow(img2)
plt.title('Image 2'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3),plt.imshow(vis)
plt.title('Matches'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4),plt.imshow(result)
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()