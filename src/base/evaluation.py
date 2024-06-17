import warnings
from typing import List

import numpy as np
import torch
from numpy import ndarray

from src.base.callable_parameters import ObjectiveFunction, GmeanObjective
from src.core.online_learning.weight_method import CostVector
from src.helper.metric import offline_metric

warnings.filterwarnings('ignore')


class Fitness:
    def __init__(self, n_class: int, func: ObjectiveFunction = GmeanObjective):
        self.n_class = n_class
        self.func = func

    def fitness(self, probs: ndarray, labels: ndarray, vector: ndarray, metrics_out: bool = False):
        assert probs.ndim == 2
        assert labels.ndim == 1
        assert vector.ndim == 1
        assert probs.shape[1] == len(vector)
        predicts = (probs * vector).argmax(axis=1)
        metrics = offline_metric(labels, predicts, self.n_class)
        if metrics_out:
            return metrics
        return self.func(*metrics)


class FitnessGmean:
    def __init__(self, n_class: int):
        self.n_class = n_class

    def gmean_for_loop(self, labels, predictions):
        label_set = np.arange(0, self.n_class)
        recall = np.zeros(len(label_set))

        for i, label in enumerate(label_set):
            a = labels == label
            b = predictions == label
            true_positive = np.sum(a & b)
            actual_positives = np.sum(a)

            if actual_positives > 0:
                recall[i] = true_positive / actual_positives

        gm = recall.prod() ** (1 / self.n_class)
        return gm

    def gmean_vector(self, labels, predictions):
        true_positive_matrix = (labels[:, np.newaxis] == np.arange(self.n_class))
        actual_positive_matrix = (labels[:, np.newaxis] == np.arange(self.n_class))

        true_positives = np.sum(true_positive_matrix & (predictions[:, np.newaxis] == np.arange(self.n_class)), axis=0)
        actual_positives = np.sum(actual_positive_matrix, axis=0)

        recall = np.where(actual_positives > 0, true_positives / actual_positives, 0.0)

        gm = recall.prod() ** (1 / self.n_class)
        return gm

    def fitness(self, probs: ndarray, labels: ndarray, vector: ndarray):
        assert probs.ndim == 2
        assert labels.ndim == 1
        assert vector.ndim == 1
        assert probs.shape[1] == len(vector)
        predicts = (probs * vector).argmax(axis=1)
        metrics = self.gmean_for_loop(labels, predicts)
        return metrics


def fitness(n_class: int, probs: List[ndarray], labels: List[ndarray],
            weighting: CostVector, metrics_out: bool = False,
            func: ObjectiveFunction = GmeanObjective):
    probs = torch.tensor(probs)
    labels = torch.tensor(labels)
    weighted_probs = weighting(probs)
    predicts = weighted_probs.argmax(axis=1)
    metrics = offline_metric(labels, predicts, n_class)
    if metrics_out:
        return metrics

    return func(*metrics)


if __name__ == '__main__':
    ...
