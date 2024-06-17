import os
import random

import numpy as np
import numpy.random
import torch
from numpy import ndarray

current_path = os.path.dirname(__file__)


def cost_vector_design(class_sizes: ndarray) -> ndarray:
    vector = 1 / (class_sizes + 0.1)
    return vector / vector.sum()


def set_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def phenotype_to_chromosome(matrix: ndarray, flexible_diagonal: bool):
    if not flexible_diagonal:
        n = len(matrix)
        matrix = matrix[~np.eye(n, dtype=bool)]
    return matrix.flatten()


def chromosome_to_phenotype(n: int, matrix: ndarray, flexible_diagonal: bool):
    if flexible_diagonal:
        return matrix.reshape(n, n)
    matrix = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = matrix[k]
                k += 1

    return matrix


if __name__ == '__main__':
    ...
