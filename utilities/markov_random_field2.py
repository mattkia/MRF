import numpy as np
import networkx as nx
import cv2
import sys

from sklearn.naive_bayes import GaussianNB
from .image_utils import IntegerSimulatedAnnealing


class MRF:
    __beta = None
    __original_image = None
    __gray_scale_image = None
    __test_image = None
    __test_gray_scale_image = None
    __original_shape = None
    __original_labels = None
    __unique_labels = None
    __mean_values = None
    __variance_values = None
    __iteration = None
    __predicted_labels = None
    __segmented_image = None
    __image_graph = None
    __neighborhood = None

    def __init__(self, original_image, original_labels, neighborhood='four'):
        self.__original_image = original_image
        self.__original_labels = original_labels
        self.__neighborhood = neighborhood

        self.__gray_scale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        self.__mean_values = []
        self.__variance_values = []

        self.__unique_labels = list(np.unique(original_labels))

        self.__original_shape = original_image.shape

        self.__calculate_mean_values()
        self.__calculate_variance_values()

    def __create_network_graph(self, neighborhood='four'):
        if neighborhood == 'four':
            print('[*] Creating Network Graph With Four-Neighborhood')
        else:
            print('[*] Creating Network Graph With Eight-Neighborhood')

        self.__image_graph = nx.Graph()

        if neighborhood == 'four':
            for i in range(self.__test_image.shape[0]):
                for j in range(self.__test_image.shape[1]):
                    self.__image_graph.add_node((i, j))
                    self.__image_graph.nodes[i, j]['intensity'] = self.__test_image[i, j]

            for i in range(self.__test_image.shape[0]):
                for j in range(self.__test_image.shape[1]):
                    if j == self.__test_image.shape[1] - 1 and i != self.__test_image.shape[0] - 1:
                        self.__image_graph.add_edge((i, j), (i + 1, j))
                    elif i == self.__test_image.shape[0] - 1 and j != self.__test_image.shape[1] - 1:
                        self.__image_graph.add_edge((i, j), (i, j + 1))
                    elif i != self.__test_image.shape[0] - 1 and j != self.__test_image.shape[1] - 1:
                        self.__image_graph.add_edge((i, j), (i, j + 1))
                        self.__image_graph.add_edge((i, j), (i + 1, j))
                    else:
                        continue
        elif neighborhood == 'eight':
            for i in range(self.__test_image.shape[0]):
                for j in range(self.__test_image.shape[1]):
                    self.__image_graph.add_node((i, j))
                    self.__image_graph.nodes[i, j]['intensity'] = self.__test_image.shape[i, j]

            for i in range(self.__test_image.shape[0]):
                for j in range(self.__test_image.shape[1]):
                    if j == self.__test_image.shape[1] - 1 and i != self.__test_image.shape[0] - 1:
                        self.__image_graph.add_edge((i, j), (i + 1, j))
                    elif i == self.__test_image.shape[0] - 1 and j != self.__test_image.shape[1] - 1:
                        self.__image_graph.add_edge((i, j), (i, j + 1))
                    elif i != self.__test_image.shape[0] - 1 and j != self.__test_image.shape[1] - 1:
                        self.__image_graph.add_edge((i, j), (i, j + 1))
                        self.__image_graph.add_edge((i, j), (i + 1, j))
                        self.__image_graph.add_edge((i, j), (i + 1, j + 1))

                    if i >= 1 and j < self.__test_image[1] - 1:
                        self.__image_graph.add_edge((i, j), (i - 1, j + 1))

        else:
            raise Exception('[!] Wrong Neighborhood Option - Choose "four" or "eight"')

    def __calculate_mean_values(self):
        print('[*] Calculating Mean Values For Each Label And Feature')
        for label in self.__unique_labels:
            means = []
            for feature in range(4):
                mean = 0
                mean_count = 0
                if feature == 0:
                    for i in range(self.__original_shape[0]):
                        for j in range(self.__original_shape[1]):
                            if self.__original_labels[i, j] == label:
                                mean += self.__gray_scale_image[i, j]
                                mean_count += 1
                    means.append(mean/mean_count)
                else:
                    for i in range(self.__original_shape[0]):
                        for j in range(self.__original_shape[1]):
                            if self.__original_labels[i, j] == label:
                                mean += self.__original_image[i, j][feature-1]
                                mean_count += 1
                    means.append(mean/mean_count)
            self.__mean_values.append(means)

    def __calculate_variance_values(self):
        print('[*] Calculating Variance Values For Each Label And Feature')
        label_index = 0
        for label in self.__unique_labels:
            variances = []
            for feature in range(4):
                if feature == 0:
                    variance = 0
                    variance_count = 0
                    for i in range(self.__original_shape[0]):
                        for j in range(self.__original_shape[1]):
                            if self.__original_labels[i, j] == label:
                                variance += (self.__gray_scale_image[i, j] - self.__mean_values[label_index][0])**2
                                variance_count += 1
                    variances.append(variance/variance_count)
                else:
                    variance = 0
                    variance_count = 0
                    for i in range(self.__original_shape[0]):
                        for j in range(self.__original_shape[1]):
                            if self.__original_labels[i, j] == label:
                                variance += (self.__original_image[i, j][feature-1] -
                                             self.__mean_values[label_index][feature-1])**2
                                variance_count += 1
                    variances.append(variance/variance_count)

            self.__variance_values.append(variances)
            label_index += 1

    def __calculate_mean_matrix(self, labels_matrix):
        means = np.zeros((4, labels_matrix.shape[0], labels_matrix.shape[1]))

        for k in range(4):
            for i in range(labels_matrix.shape[0]):
                for j in range(labels_matrix.shape[1]):
                    means[k, i, j] = self.__mean_values[self.__unique_labels.index(labels_matrix[i, j])][k]

        return means

    def __calculate_variance_matrix(self, labels_matrix):
        variances = np.zeros((4, labels_matrix.shape[0], labels_matrix.shape[1]))

        for k in range(4):
            for i in range(labels_matrix.shape[0]):
                for j in range(labels_matrix.shape[1]):
                    variances[k, i, j] = self.__variance_values[self.__unique_labels.index(labels_matrix[i, j])][k]

        return variances

    def __energy_function(self, label_matrix):
        energy_term2 = 0

        new_label_matrix = label_matrix.reshape(self.__test_image.shape[0], self.__test_image.shape[1])

        variances = self.__calculate_variance_matrix(new_label_matrix)

        means = self.__calculate_mean_matrix(new_label_matrix)

        term1 = 0
        term2 = 0
        for k in range(4):
            term1 += np.log(np.sqrt(2 * np.pi) * variances[k])
            if k == 0:
                term2 += ((self.__test_gray_scale_image - means[0]) ** 2) / (2 * variances[0])
            else:
                term2 += ((self.__test_image[:, :, k-1] - means[k]) ** 2) / (2 * variances[k])

        energy_term1 = np.sum(term1 + term2)

        for i in range(self.__original_shape[0]):
            for j in range(self.__original_shape[1]):
                for neighbor in self.__image_graph[i, j]:
                    if new_label_matrix[i, j] == new_label_matrix[neighbor]:
                        energy_term2 -= self.__beta
                    else:
                        energy_term2 += self.__beta

        energy = energy_term1 + energy_term2

        return energy

    def __update_energy(self, previous_energy, target_index, new_label, old_label, current_state):
        old_variance = self.__variance_values[old_label]
        old_mean = self.__mean_values[old_label]

        new_variance = self.__variance_values[new_label]
        new_mean = self.__mean_values[new_label]

        updater1 = 0
        updater2 = 0

        for k in range(4):
            if k == 0:
                updater1 += -np.log(np.sqrt(2*np.pi)*old_variance[k]) - \
                            ((self.__test_gray_scale_image[target_index] - old_mean[k])**2) / (2*old_variance[k])
                updater2 += np.log(np.sqrt(2*np.pi)*new_variance[k]) + \
                            ((self.__test_gray_scale_image[target_index] - new_mean[k])**2) / (2*new_variance[k])
            else:
                updater1 += -np.log(np.sqrt(2 * np.pi) * old_variance[k]) - \
                            ((self.__test_image[target_index][k-1] - old_mean[k]) ** 2) / (2 * old_variance[k])
                updater2 += np.log(np.sqrt(2 * np.pi) * new_variance[k]) + \
                            ((self.__test_image[target_index][k-1] - new_mean[k]) ** 2) / (2 * new_variance[k])

        new_energy_term1 = updater1 + updater2

        new_energy_term2 = 0

        for neighbor in self.__image_graph[target_index]:
            if old_label == current_state[neighbor]:
                new_energy_term2 += 4 * self.__beta
            else:
                if new_label == current_state[neighbor]:
                    new_energy_term2 -= 4 * self.__beta

        new_energy = previous_energy + new_energy_term1 + new_energy_term2

        return new_energy

    def segment(self, test_image, beta=1, mode='random'):
        self.__beta = beta
        self.__test_image = test_image
        self.__create_network_graph(neighborhood=self.__neighborhood)

        self.__test_gray_scale_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        if mode == 'random':
            print('[*] Random Initialization')
            initial_labels = np.random.randint(0, 3, size=self.__test_image.shape)
            max_iterations = 4000000
        else:
            print('[*] Initializing From Naive Bayes')
            clf = GaussianNB()
            clf.fit(self.__original_image.reshape(self.__original_shape[0] * self.__original_shape[1], 3),
                    self.__original_labels.reshape(self.__original_shape[0] * self.__original_shape[1],))

            initial_labels = clf.predict(self.__test_image.reshape(self.__test_image.shape[0] * self.__test_image.shape[1], 3)).reshape(self.__test_image.shape[0], self.__test_image.shape[1]).astype(np.int)

            max_iterations = 500000

        lower_bound = 0
        upper_bound = 2

        sim_ann = IntegerSimulatedAnnealing(self.__energy_function,
                                            self.__update_energy,
                                            initial_labels,
                                            lower_bound,
                                            upper_bound,
                                            temperature=25000)

        sim_ann.fit(max_iteration=max_iterations)

        predicted_labels = sim_ann.get_minimizer()

        recast_labels = predicted_labels.reshape(self.__test_image.shape[0], self.__test_image.shape[1])

        return recast_labels




