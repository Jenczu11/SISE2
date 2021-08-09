import numpy as np

from config import NEURAL_NETWORK_SETTINGS


def norm(dataset):
    if NEURAL_NETWORK_SETTINGS.NORMALIZATION:
        max = abs(dataset.values.max())
        min = abs(dataset.values.min())
        print(max)
        print(min)
        divider = np.maximum(min,max)
        dataset = dataset.div(NEURAL_NETWORK_SETTINGS.NORMALIZATION_DIVIDER)
    return dataset


def unnorm(dataset):
    if NEURAL_NETWORK_SETTINGS.NORMALIZATION:
        dataset = dataset.mul(NEURAL_NETWORK_SETTINGS.NORMALIZATION_DIVIDER)
    return dataset