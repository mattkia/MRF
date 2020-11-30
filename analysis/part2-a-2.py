import numpy as np
import cv2
import sys

from ..utilities.image_utils import ImageUtils
from ..utilities.markov_random_field import MRF


##############################################
# Reading The Original Image
##############################################
print('[*] Reading Image')
original_image = cv2.imread('../../images/test1.bmp', 0)
original_shape = original_image.shape
color_image = cv2.imread('../../images/test2.jpg')
color_shape = (color_image.shape[0], color_image.shape[1])

hsv_transform = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
hsv_image = np.zeros(color_shape, dtype=np.uint8)
for i in range(color_shape[0]):
    for j in range(color_shape[1]):
        hsv_image[i, j] = hsv_transform[i, j][0]


##############################################
# Creating a Label Matrix
##############################################

print('[*] Creating The Label Matrix Based on the Intensities of the Original Image')
labels = np.zeros(original_shape)
for row in range(original_shape[0]):
    for column in range(original_shape[1]):
        if original_image[row, column] == 0:
            labels[row, column] = 0
        elif original_image[row, column] == 127:
            labels[row, column] = 1
        elif original_image[row, column] == 255:
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
res = mrf.segment(hsv_image, mode='nb')

recast_image = np.zeros(color_shape, dtype=np.uint8)
for i in range(color_shape[0]):
    for j in range(color_shape[1]):
        if res[i, j] == 0:
            recast_image[i, j] = 0
        elif res[i, j] == 1:
            recast_image[i, j] = 127
        elif res[i, j] == 2:
            recast_image[i, j] = 255

cv2.imwrite('../results/part2-a/mrf2.png', recast_image)



