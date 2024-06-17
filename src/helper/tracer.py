import math
from copy import deepcopy

import numpy as np
from numpy import ndarray
from torch import Tensor

from src.base.configuration import Configuration
from src.helper.helper import cost_vector_design
from src.helper.metric import precision_recall_fscore_support_by_confusion_matrix


class CITracer:
    def __init__(self, config: Configuration):
        self.config = config
        self.n_class = config.dataset_meta.n_class
        self.class_sizes = np.zeros(shape=self.n_class)
        self.fading_factor = config.online_learning_meta.fading_factor

        self.overall_ir = np.max(self.config.dataset_meta.class_sizes) / (
                1e-3 + np.min(self.config.dataset_meta.class_sizes))

    def update(self, label: int):
        # update class size
        self.class_sizes = self.class_sizes * self.fading_factor
        self.class_sizes[label] += 1

    def ir2(self, clip: bool = True):
        assert len(self.class_sizes) == 2
        imbalance_ratio = self.class_sizes[0] / (1e-3 + self.class_sizes[1])
        if clip:
            return min(imbalance_ratio, self.overall_ir * 2)
        return imbalance_ratio

    def ir(self, clip: bool = True):
        imbalance_ratio = np.max(self.class_sizes) / (1e-3 + np.min(self.class_sizes))
        if clip:
            return min(imbalance_ratio, self.overall_ir * 2)
        return imbalance_ratio

    def cost_vector(self):
        """
        :return: best vector for naive method
        """
        mode = self.config.comparison_mode
        if mode == 'ir':
            return cost_vector_design(self.class_sizes)
        elif mode == 'overall_ir':
            if hasattr(self, 'overall_ir_vector'):
                return self.overall_ir_vector
            self.overall_ir_vector = cost_vector_design(self.config.dataset_meta.class_sizes)
            return self.overall_ir_vector
        elif mode == 'grid':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def cost_candidates(self, n: int = None):
        if n is None:
            if self.config.optimization_method == 'soea':
                n = self.config.soea_meta.n_prior_individual
            elif self.config.optimization_method == 'moea':
                n = self.config.moea_meta.n_prior_individual
            elif self.config.optimization_method == 'grid':
                n = self.config.grid_meta.n_prior_individual
            elif self.config.optimization_method == 'no':
                n = 1
            else:
                raise NotImplementedError()

        refer = self.cost_vector()

        if self.config.init_mode == 'disturbance':
            candidates = self.cost_vectors_disturbance(refer, n, radius=0.1)
        elif self.config.init_mode == 'grid':
            candidates = self.cost_vectors_grid(n)
        elif self.config.init_mode == 'exact':
            candidates = [deepcopy(refer) for _ in range(n)]
        elif self.config.init_mode == 'random':
            candidates = [np.random.random(refer.shape) for _ in range(n)]
            candidates = [candidate / candidate.sum() for candidate in candidates]
        else:
            raise NotImplementedError()
        return candidates

    def cost_vectors_disturbance(self, refer: ndarray, n: int = None, radius: float = 1):
        """
        :return: candidate vectors evolutionary algorithm
        """
        factors = np.linspace(0, radius, n)
        candidates = []
        for factor in factors:
            vector = refer + (np.random.rand(*refer.shape) * 2 - 1) * factor
            vector[vector < 0] = 0
            candidates.append(vector)
        return [candidate / candidate.sum() for candidate in candidates]

    def cost_vectors_grid(self, n: int):
        candidates = []
        for i in range(1, 1 + n // 2):
            candidates.append(np.array([1, i]))
            candidates.append(np.array([i, 1]))
        if len(candidates) < n:
            candidates.append(np.array([1, 1]))
        return [candidate / candidate.sum() for candidate in candidates]


@DeprecationWarning
class MetricTracerConfusionMatrix:
    def __init__(self, config: Configuration = None, n_class=None):
        self.config = config
        if config is None:
            self.n_class = n_class
        else:
            self.n_class = config.dataset_meta.n_class
        self.confusion_matrix = np.zeros(shape=(self.n_class, self.n_class))
        self.fading_factor = 0.99

        (self.accuracy_li, self.gmean_li, self.precision_li, self.recall_li,
         self.fscore_li, self.support_li, self.confusion_li, self.balanced_accuracy_li) = [], [], [], [], [], [], [], []

    def update(self, predict: int, label: int):
        # update confusion matrix
        self.confusion_matrix = self.confusion_matrix * self.fading_factor
        self.confusion_matrix[label, predict] += 1

        # append metrics
        accuracy, gmean, precision, recall, fscore, support = precision_recall_fscore_support_by_confusion_matrix(
            self.confusion_matrix)
        self.accuracy_li.append(accuracy)
        self.gmean_li.append(gmean)
        self.precision_li.append(precision)
        self.recall_li.append(recall)
        self.fscore_li.append(fscore)
        self.support_li.append(support)
        self.confusion_li.append(self.confusion_matrix)

    # about online metric
    def instantaneous_metric(self):
        return self.accuracy_li[-1], self.gmean_li[-1], self.precision_li[-1], self.recall_li[-1], self.fscore_li[-1], \
            self.support_li[-1], self.confusion_li[-1]

    def instantaneous_gmean(self):
        return self.gmean_li[-1]

    def overall_metric(self):
        return self._means(self.accuracy_li, self.gmean_li, self.precision_li, self.recall_li, self.fscore_li,
                           self.support_li, self.confusion_li)

    def overall_gmean(self):
        return self._means(self.gmean_li)

    def metric_list(self):
        return self.accuracy_li, self.gmean_li, self.precision_li, self.recall_li, self.fscore_li, self.support_li, self.confusion_li

    @staticmethod
    def _means(*args):
        li = []
        for arg in args:
            li.append(np.nanmean(np.array(arg), axis=0))
        return li


class MetricTracer:
    def __init__(self, n_class: int):
        self.n_class = n_class
        self.fading_factor = 0.99

        self.s, self.n = np.zeros([self.n_class]), np.zeros([self.n_class])
        self.gmean_li = []
        self.balanced_accuracy_li = []
        self.recall_li = []

    def update(self, predict: int, label: int):
        recall, gmean, balanced_accuracy, self.s, self.n = pf_online(self.s, self.n, label, predict,
                                                                     theta=self.fading_factor)
        self.recall_li.append(recall)
        self.gmean_li.append(gmean)
        self.balanced_accuracy_li.append(balanced_accuracy)

    # about online metric
    def instantaneous_metric(self):
        return self.gmean_li[-1], self.balanced_accuracy_li[-1], self.recall_li[-1]

    def instantaneous_gmean(self):
        return self.gmean_li[-1]

    def overall_gmean(self):
        return self._means(self.gmean_li)

    def overall_metric(self):
        return self._means(self.gmean_li, self.balanced_accuracy_li, self.recall_li)

    @staticmethod
    def _means(*args):
        li = []
        for arg in args:
            li.append(np.nanmean(np.array(arg), axis=0))
        return li


def get_online_performance_predict(labels, predicts):
    assert len(labels) == len(predicts)
    tracer = MetricTracer(n_class=len(np.unique(labels)))
    for p, l in zip(predicts, labels):
        tracer.update(p, l)
    gmean, balanced_accuracy, _ = tracer.overall_metric()
    return gmean, balanced_accuracy, tracer.gmean_li, tracer.balanced_accuracy_li


def get_online_performance_ensemble(labels, probs, vectors):
    tracer = MetricTracer(n_class=len(np.unique(labels)))
    # tracer = MetricTracerConfusionMatrix(n_class=len(np.unique(labels)))
    for l, p, v in zip(labels, probs, vectors):
        if isinstance(v[0], Tensor):
            v = [np.array(a) for a in v]
        v_ = np.array(v).mean(axis=0)
        predict = (v_ * p).argmax()
        label = np.array(l).item()
        tracer.update(predict, label)
    gmean, balanced_accuracy, _ = tracer.overall_metric()
    # balanced_accuracy, gmean, _, _, _, _, _ = tracer.overall_metric()
    return gmean, balanced_accuracy, tracer.gmean_li, tracer.balanced_accuracy_li


def get_online_performance_single(labels, probs, vectors):
    tracer = MetricTracer(n_class=len(np.unique(labels)))
    # tracer = MetricTracerConfusionMatrix(n_class=len(np.unique(labels)))
    for l, p, v in zip(labels, probs, vectors):
        if isinstance(v[0], Tensor):
            v = [np.array(a) for a in v]
        v_ = np.array(v)[0]
        predict = (v_ * p).argmax()
        label = np.array(l).item()
        tracer.update(predict, label)
    gmean, balanced_accuracy, _ = tracer.overall_metric()
    # balanced_accuracy, gmean, _, _, _, _, _ = tracer.overall_metric()
    return gmean, balanced_accuracy, tracer.gmean_li, tracer.balanced_accuracy_li


def get_online_performance_naive(labels, probs, vectors):
    tracer = MetricTracer(n_class=len(np.unique(labels)))
    # tracer = MetricTracerConfusionMatrix(n_class=len(np.unique(labels)))
    for l, p, v in zip(labels, probs, vectors):
        if isinstance(v[0], Tensor):
            v = [np.array(a) for a in v]
        predict = (v * p).argmax()
        label = np.array(l).item()
        tracer.update(predict, label)
    gmean, balanced_accuracy, _ = tracer.overall_metric()
    # balanced_accuracy, gmean, _, _, _, _, _ = tracer.overall_metric()
    return gmean, balanced_accuracy, tracer.gmean_li, tracer.balanced_accuracy_li


def uniform_chromosome(n_class, n_individual):
    chrom = np.ones(shape=n_class)
    chrom = chrom / chrom.sum()
    chroms = np.atleast_2d(chrom).repeat(n_individual, axis=0)
    return chroms


def pf_online(S, N, y_t, p_t, theta=0.99):
    c = int(y_t)
    S[c] = (y_t == p_t) + theta * (S[c])
    N[c] = 1 + theta * N[c]

    recall = S / N
    gmean = gmean_compute(recall)
    balanced_accuracy = balanced_accuracy_compute(recall)
    return recall, gmean, balanced_accuracy, S, N


def gmean_compute(recall):
    gmean = 1
    n = 0
    for r in recall:
        if math.isnan(r):
            n = n + 1
        else:
            gmean = gmean * r
    gmean = pow(gmean, 1 / (len(recall) - n))
    return gmean


def balanced_accuracy_compute(recall):
    return np.nanmean(recall)


if __name__ == '__main__':
    ...
