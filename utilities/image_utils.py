import numpy as np
import sys


class ImageUtils:
    __image = None

    GAUSSIAN_NOISE = 0
    SALT_AND_PEPPER_NOISE = 1
    POISSON_NOISE = 2
    SPECKLE_NOISE = 3

    def __init__(self, img):
        self.__image = img

    def insert_noise(self, noise_type, gaussian_variance=0.01):
        if noise_type == 0:
            if len(self.__image.shape) == 2:
                row, col = self.__image.shape
                mean = 0
                var = gaussian_variance
                sigma = np.sqrt(var)
                gauss = np.random.normal(mean, sigma, (row, col))
                gauss = gauss.reshape(row, col)
                noisy = self.__image + gauss
                for i in range(noisy.shape[0]):
                    for j in range(noisy.shape[1]):
                        if noisy[i, j] < 0:
                            noisy[i, j] = 0
                        elif noisy[i, j] > 255:
                            noisy[i, j] = 255
                return noisy.astype(np.uint8)
            else:
                row, col, ch = self.__image.shape
                mean = 0
                var = gaussian_variance
                sigma = np.sqrt(var)
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                noisy = self.__image + gauss
                for i in range(noisy.shape[0]):
                    for j in range(noisy.shape[1]):
                        for k in range(noisy.shape[2]):
                            if noisy[i, j, k] < 0:
                                noisy[i, j, k] = 0
                            elif noisy[i, j, k] > 255:
                                noisy[i, j, k] = 255
                return noisy.astype(np.uint8)
        elif noise_type == 1:
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(self.__image)
            num_salt = np.ceil(amount * self.__image.size * s_vs_p)
            coordinates = [np.random.randint(0, i-1, int(num_salt)) for i in self.__image.shape]
            out[coordinates] = 1
            num_peppers = np.ceil(amount * self.__image.size * (1 - s_vs_p))
            coordinates = [np.random.randint(0, i-1, int(num_peppers)) for i in self.__image.shape]
            out[coordinates] = 0
            return out
        elif noise_type == 2:
            values = len(np.unique(self.__image))
            values = 2 ** np.ceil(np.log2(values))
            noisy = np.random.poisson(self.__image * values) / float(values)
            return noisy
        elif noise_type == 3:
            row, col, ch = self.__image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = self.__image + self.__image * gauss
            return noisy

    def vectorize(self):
        out = self.__image.reshape(self.__image.shape[0]*self.__image.shape[1], 1)
        return out


class IntegerSimulatedAnnealing:
    __objective_function = None
    __updater = None
    __initial_state = None
    __lower_bound = None
    __upper_bound = None
    __temperature = None
    __current_state = None
    __local_minimum = None
    __minimizer = None
    __chosen_neighbor_energy = None
    __current_state_energy = None

    def __init__(self, objective_function, updater, initial_state, lower_bound, upper_bound, temperature=20000):
        self.__objective_function = objective_function
        self.__updater = updater
        self.__initial_state = initial_state
        self.__current_state = initial_state
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__temperature = temperature

    def __randomly_choose_neighbor(self):
        random_i = np.random.randint(0, self.__current_state.shape[0])
        random_j = np.random.randint(0, self.__current_state.shape[1])

        neighbor = np.copy(self.__current_state)

        random_value = np.random.randint(self.__lower_bound, self.__upper_bound + 1)

        while neighbor[random_i, random_j] == random_value:
            random_value = np.random.randint(self.__lower_bound, self.__upper_bound + 1)

        neighbor[random_i, random_j] = random_value

        return random_value, (random_i, random_j)

    def __check_acceptability(self, old_energy, chosen_neighbor):
        new_energy = self.__updater(old_energy, chosen_neighbor[1], chosen_neighbor[0],
                                    self.__current_state[chosen_neighbor[1]], self.__current_state)

        self.__chosen_neighbor_energy = new_energy

        if new_energy < old_energy:
            return True
        return False

    def __move(self, chosen_neighbor):
        acceptable = self.__check_acceptability(self.__current_state_energy, chosen_neighbor)

        if acceptable is True:
            self.__current_state[chosen_neighbor[1]] = chosen_neighbor[0]
            self.__current_state_energy = self.__chosen_neighbor_energy
        else:
            probability = np.exp(-(self.__chosen_neighbor_energy - self.__current_state_energy) / self.__temperature)

            random_number = np.random.rand()

            if random_number <= probability:
                self.__current_state[chosen_neighbor[1]] = chosen_neighbor[0]
                self.__current_state_energy = self.__chosen_neighbor_energy

    def fit(self, max_iteration=10000):
        print('[*] Cooling Down The System')
        self.__current_state_energy = self.__objective_function(self.__initial_state)
        for iteration in range(max_iteration):
            print('[*] Simulated Annealing Iteration : ', iteration + 1, ', Temperature Is :', self.__temperature,
                  'Total Energy : ', self.__current_state_energy)
            chosen_neighbor = self.__randomly_choose_neighbor()
            self.__move(chosen_neighbor)
            if iteration + 1 > 2:
                if self.__temperature / np.log(iteration) >= 0.1:
                    self.__temperature = self.__temperature / np.log(iteration)
                    # self.__temperature = 0.9 * self.__temperature
                else:
                    self.__temperature = 0.1

        self.__local_minimum = self.__current_state_energy
        self.__minimizer = self.__current_state

        print('[*] System Is Frozen - Optimization Finished')

    def get_minimizer(self):
        return self.__minimizer

    def get_local_minimum(self):
        return self.__local_minimum





