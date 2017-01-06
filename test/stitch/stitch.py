from stitcher import Stitcher
import argparse
import imutils
import cv2

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

cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)