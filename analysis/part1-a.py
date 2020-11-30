import cv2
import numpy as np

from sklearn.naive_bayes import GaussianNB
from ..utilities.image_utils import ImageUtils


##############################################
# Reading The Original Image
##############################################
print('[*] Reading Image')
original_image = cv2.imread('../../images/test1.bmp', 0)
original_shape = original_image.shape


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
noisy_image = image_utility.insert_noise(ImageUtils.GAUSSIAN_NOISE, gaussian_variance=1000000)


##############################################
# Learning the parameters of the Classifier
##############################################

print('[*] Creating the Image Vector')
image_utility = ImageUtils(noisy_image)
noisy_image_vector = image_utility.vectorize()

print('[*] Creating the Labels Vector')
image_utility = ImageUtils(labels)
labels_vector = image_utility.vectorize()
labels_vector = labels_vector.reshape(labels_vector.shape[0],)

print('[*] Training the Gaussian Naive Bayes Classifier')
classifier = GaussianNB()
classifier.fit(noisy_image_vector, labels_vector)
print(classifier.get_params())
print('[*] Naive Bayes Classifier Trained Successfully')


##############################################
# Predicting the Labels of the Original Image
##############################################

print('[*] Classifying the Original Image')
image_utility = ImageUtils(original_image)
original_image_vector = image_utility.vectorize()
predicted_vector = classifier.predict(original_image_vector)
print('[*] Classification Terminated Successfully')


##############################################
# Calculating the Accuracy of the Classifier
##############################################

print('[*] Calculating the Accuracy of the Gaussian Naive Bayes Classifier')
mis_classifications = 0
for i in range(len(labels_vector)):
    if labels_vector[i] != predicted_vector[i]:
        mis_classifications += 1
print('[*] Number of miscalssified pixels : ', mis_classifications)


#######################################################
# Recasting the Classified Image to the Original Shape
#######################################################

recast_labels = predicted_vector.reshape(original_shape)
recast_image = np.zeros(original_shape, dtype=np.uint8)
for i in range(original_shape[0]):
    for j in range(original_shape[1]):
        if recast_labels[i, j] == 0:
            recast_image[i, j] = 0
        elif recast_labels[i, j] == 1:
            recast_image[i, j] = 127
        elif recast_labels[i, j] == 2:
            recast_image[i, j] = 255


total_image = np.concatenate((original_image, noisy_image, recast_image), axis=1)

cv2.imwrite('../results/part1-a-b/var1000000.png', total_image)

##############################################
##############################################
##############################################

