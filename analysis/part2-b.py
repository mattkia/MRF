import numpy as np
import cv2

from ..utilities.image_utils import ImageUtils
from ..utilities.markov_random_field2 import MRF

##############################################
# Reading The Original Image
##############################################
print('[*] Reading Image')
original_image = cv2.imread('../../images/test3.jpg')
original_shape = original_image.shape

target_image = cv2.imread('../../images/test2.jpg')
target_shape = target_image.shape

##############################################
# Creating a Label Matrix
##############################################

print('[*] Creating The Label Matrix Based on the Intensities of the Original Image')
labels = np.zeros((original_shape[0], original_shape[1])).astype(np.int)
for row in range(original_shape[0]):
    for column in range(original_shape[1]):
        if (original_image[row, column] == [235, 165, 114]).all():
            labels[row, column] = 0
        elif (original_image[row, column] == [31, 100, 55]).all():
            labels[row, column] = 1
        elif (original_image[row, column] == [76, 74, 80]).all():
            labels[row, column] = 2

##############################################
# Making the Image Noisy
##############################################

print('[*] Making the Original Image Noisy')
image_utility = ImageUtils(original_image)
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

cv2.imwrite('../results/part2-b/mrf1.png', recast_image)



