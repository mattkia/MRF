import numpy as np
import cv2

from ..utilities.image_utils import ImageUtils
from ..utilities.markov_random_field2 import MRF

##############################################
# Reading The Original Image
##############################################
print('[*] Reading Image')
label_image = cv2.imread('../../images/test4.jpg', cv2.IMREAD_GRAYSCALE)
label_shape = label_image.shape

target_image = cv2.imread('../../images/test2.jpg')
target_shape = target_image.shape

##############################################
# Creating a Label Matrix
##############################################

print('[*] Creating The Label Matrix Based on the Intensities of the Original Image')
labels = np.zeros(label_shape).astype(np.int)
for row in range(label_shape[0]):
    for column in range(label_shape[1]):
        if label_image[row, column] == 253 or label_image[row, column] == 254 or label_image[row, column] == 255:
            labels[row, column] = 0
        elif label_image[row, column] == 126 or label_image[row, column] == 127 \
                or label_image[row, column] == 128 or label_image[row, column] == 129 \
                or label_image[row, column] == 130:
            labels[row, column] = 1
        elif label_image[row, column] == 20 or label_image[row, column] == 21 \
                or label_image[row, column] == 22 or label_image[row, column] == 23 \
                or label_image[row, column] == 24:
            labels[row, column] = 2

##############################################
# Making the Image Noisy
##############################################

print('[*] Making the Original Image Noisy')
image_utility = ImageUtils(target_image)
noisy_image = image_utility.insert_noise(ImageUtils.GAUSSIAN_NOISE, gaussian_variance=1000)

############################################################
# Using Markov Random Field For Segmentation
############################################################

print('[*] Using Markov Random Field For Segmentation')
mrf = MRF(noisy_image, labels)
res = mrf.segment(target_image, mode='nb')

recast_image = np.zeros(target_shape, dtype=np.uint8)
for i in range(target_shape[0]):
    for j in range(target_shape[1]):
        if res[i, j] == 0:
            recast_image[i, j] = [235, 167, 114]
        elif res[i, j] == 1:
            recast_image[i, j] = [31, 100, 55]
        elif res[i, j] == 2:
            recast_image[i, j] = [76, 74, 80]

cv2.imwrite('../results/part2-b/mrf2.png', recast_image)



