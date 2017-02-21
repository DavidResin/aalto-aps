import cv2
import numpy as np

class Stitcher:
	def ransac(self, descr1, descr2, ratio=0.75, reprojThresh=4.0):
		(kp1, feat1) = descr1
		(kp2, feat2) = descr2
		return self.matchKeypoints(kp1, kp2, feat1, feat2, ratio, reprojThresh)

	def stitch(self, imgs, ratio=0.75, reprojThresh=4.0, showMatches=False):
		(img2, img1) = imgs
		descr1 = self.detectAndDescribe(img1)
		descr2 = self.detectAndDescribe(img2)
		(kp1, feat1) = descr1
		(kp2, feat2) = descr2

		ransac = self.ransac(descr1, descr2, ratio, reprojThresh)

		if ransac is None:
			return None

		(matches, matrix, status) = ransac
		result = cv2.warpPerspective(img1, matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))
		result[0:img2.shape[0], 0:img2.shape[1]] = img2

		if showMatches:
			vis = self.drawMatches(img1, img2, kp1, kp2, matches, status)
			return (result, vis)

		return result

	def detectAndDescribe(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		descriptor = cv2.xfeatures2d.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)

		kps = np.float32([kp.pt for kp in kps])

		return (kps, features)

	def matchKeypoints(self, kp1, kp2, feat1, feat2, ratio, reprojThresh):
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(feat1, feat2, 2)
		matches = []

		for m in rawMatches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		if len(matches) > 4:
			pts1 = np.float32([kp1[i] for (_, i) in matches])
			pts2 = np.float32([kp2[i] for (i, _) in matches])

			(matrix, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)

			return (matches, matrix, status)

		return None

	def drawMatches(self, img1, img2, kp1, kp2, matches, status):
		(h1, w1) = img1.shape[:2]
		(h2, w2) = img2.shape[:2]

		vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
		vis[0:h1, 0:w1] = img1
		vis[0:h2, w1:] = img2

		for ((trainIdx, queryIdx), s) in zip(matches, status):
			if s == 1:
				pt1 = (int (kp1[queryIdx][0]), int (kp1[queryIdx][1]))
				pt2 = (int (kp2[trainIdx][0]) + w1, int (kp2[trainIdx][1]))
				cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

		return vis