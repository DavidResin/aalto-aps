import cv2
import numpy as np

# These functions take care of the feature detection and transformation extraction

# Returns the RANSAC data from the given descriptors
def ransac(descr1, descr2, ratio=0.75, reprojThresh=4.0):
	(kp1, feat1) = descr1
	(kp2, feat2) = descr2
	return matchKeypoints(kp1, kp2, feat1, feat2, ratio, reprojThresh)

# Returns SIFT keypoints and features from the given image
def detectAndDescribe(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	descriptor = cv2.xfeatures2d.SIFT_create()
	(kps, features) = descriptor.detectAndCompute(image, None)

	kps = np.float32([kp.pt for kp in kps])

	return (kps, features)

# Matches 2 sets of keypoints, returning the list of matches, the corresponding homography and a status value that confirms everything went well
def matchKeypoints(kp1, kp2, feat1, feat2, ratio=0.75, reprojThresh=4.0):
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(feat1, feat2, 2)
	matches = []

	for m in rawMatches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			matches.append((m[0].trainIdx, m[0].queryIdx))

	if len(matches) > 4:
		(matrix, status) = getHomography(kp1, kp2, matches, reprojThresh)

		return (matches, matrix, status)

	return None

# Returns a homography from 2 sets of keypoints
def getHomography(kp1, kp2, matches, reprojThresh=4.0):
	pts1 = np.float32([kp1[i] for (_, i) in matches])
	pts2 = np.float32([kp2[i] for (i, _) in matches])

	return cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)

# Returns an affine transformation from 2 sets of keypoints
def getAffine(kp1, kp2, matches):
	pts1 = np.float32([kp1[i] for (_, i) in matches])
	pts2 = np.float32([kp2[i] for (i, _) in matches])

	return np.vstack((cv2.estimateRigidTransform(pts1, pts2, fullAffine=True), np.array([[0, 0, 1]])))