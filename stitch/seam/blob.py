import scipy, cv2
from scipy import ndimage

# read image into numpy array
# $ wget http://pythonvision.org/media/files/images/dna.jpeg
image1 = cv2.imread("tips.jpg")
dna = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) # gray-scale image


# smooth the image (to remove small objects); set the threshold
dnaf = ndimage.gaussian_filter(dna, 16)
T = 25 # set threshold by hand to avoid installing `mahotas` or
       # `scipy.stsci.image` dependencies that have threshold() functions

# find connected components
labeled, nr_objects = ndimage.label(dna > T) # `dna[:,:,0]>T` for red-dot case
print("Number of objects is %d " % nr_objects)

# show labeled image
####scipy.misc.imsave('labeled_dna.png', labeled)
####scipy.misc.imshow(labeled) # black&white image
import matplotlib.pyplot as plt
plt.imsave('labeled_dna.png', labeled)
plt.imshow(labeled)

plt.show()