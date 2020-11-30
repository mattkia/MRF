import numpy as np
from itertools import product
from scipy.stats import norm
import cv2
from matplotlib import pylab


def energy(state, beta, log_p):
    e = 0
    for i, j in product(range(1, state.shape[0]-1), range(1, state.shape[1]-1)):
        e -= log_p[state[i, j]][i, j]

        row, col = state[i, j-1:j+2], state[i-1:i+2, j]
        diffs = np.abs(np.concatenate((np.diff(row), np.diff(col))))
        diffs = np.clip(diffs, 0, 1)

        e += beta * diffs.sum()

    return e


def update_energy(e, neighbourhood, candidate, i, j, log_p, beta):
    current_label = neighbourhood[1, 1]
    e += log_p[current_label][i, j]

    row, col = neighbourhood[1, :], neighbourhood[:, 1]
    diffs = np.abs(np.concatenate((np.diff(row), np.diff(col))))
    diffs = np.clip(diffs, 0, 1)

    e -= 2 * beta * diffs.sum()

    neighbourhood[1, 1] = candidate

    e -= log_p[candidate][i, j]

    row, col = neighbourhood[1, :], neighbourhood[:, 1]
    diffs = np.abs(np.concatenate((np.diff(row), np.diff(col))))
    diffs = np.clip(diffs, 0, 1)

    e += 2 * beta * diffs.sum()

    return e


def simanneal(clean, image, means, vars, beta, init='random', Tmax=1000.0, Tmin=0.0, steps=1e6, rand=np.random.RandomState(0)):
    k = 0
    if init == 'nb':
        intensities, counts = np.unique(clean, return_counts=True)
        class_probabilities = np.log(counts) - np.log(counts.sum())
        state = naive_bayes_segmentation(image, intensities, class_probabilities)
    else:
        state = rand.randint(low=0, high=len(means), size=image.shape)
    state = np.pad(state, 1, mode='edge')
    log_p = []
    for mu, sig in zip(means, vars):
       log_p.append(np.pad(norm.logpdf(image, mu, sig), 1, mode='constant'))

    dT = (Tmax - Tmin) / steps
    T = Tmax
    Uold = energy(state, beta, log_p)
    while T > Tmin:
        if k % 5000 == 0:
            print('%d\t%1.1f' % (Uold, T))

        i = rand.randint(low=1, high=state.shape[0]-1)
        j = rand.randint(low=1, high=state.shape[1]-1)

        neighbourhood = np.copy(state[i - 1:i + 2, j - 1:j + 2])
        candidate = rand.choice(tuple({0, 1, 2} - {state[i, j]}))

        Unew = update_energy(Uold, neighbourhood, candidate, i, j, log_p, beta)
        dU = Unew - Uold
        if dU <= 0 or rand.rand() < np.exp(-dU/T):
            state[i, j] = candidate
            Uold = Unew

        T -= dT
        k += 1

    return state[1:-1, 1:-1]


if __name__ == '__main__':
    clean_img = cv2.imread('test1.bmp', cv2.IMREAD_GRAYSCALE)
    noise_var = 10
    print('Energy\t\tTemprature')
    noisy_img = generate_noisy_image(clean_img, noise_var)

    mus = np.unique(clean_img)
    sigs = np.ones_like(mus)
    beta = 10

    result = simanneal(clean_img, noisy_img, mus, sigs, beta)

    fig, ax = pylab.subplots(nrows=1, ncols=3)
    ax[0].imshow(clean_img, cmap='gray')
    ax[0].set_title('Clean')
    ax[1].imshow(noisy_img, cmap='gray')
    ax[1].set_title('Noisy')
    ax[2].imshow(result)
    ax[2].set_title('Result')

    pylab.show()