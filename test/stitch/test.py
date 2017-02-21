import image_warp, math, cv2
src = cv2.imread("guildroom/4.jpg")
import numpy as np
H = np.array([[math.cos(-.1), -math.sin(-.1), -40], [math.sin(-.1), math.cos(-.1), 0], [0, 0, 1]])
iw = image_warp.Image_Warp()
dst, trans = iw.homography_warp(src, H)