from copy import deepcopy
from typing import List

import geatpy as ea
import numpy as np
import torch
from numpy import ndarray
from sklearn.neighbors import NearestNeighbors
from skmultiflow.trees import HoeffdingTreeClassifier
from torch import optim

from src.base.callable_parameters import GmeanObjective
from src.base.configuration import Configuration, SOEAMeta, MOEAMeta, GDMeta, BOMeta, GridMeta
from src.base.cost_problem import SOEACostVectorProblem
from src.base.evaluation import Fitness
from src.helper.tracer import uniform_chromosome


class CostVectorAdaptor:
    def __init__(self, config: Configuration):
        self.config = config
        self.data_buffer, self.label_buffer = [], []
        self.n_class = self.config.dataset_meta.n_class
        self.ensemble_std = 0
        try:
            self.classifier = config.online_learning_meta.classifier()
        except:
            self.classifier = config.online_learning_meta.classifier(n_feature=config.dataset_meta.n_feature,
                                                                     n_class=config.dataset_meta.n_class,
                                                                     lr=self.config.online_learning_meta.online_classifier_lr)
        self.classifier.classes = list(map(lambda x: int(x), self.config.dataset_meta.label_set))

        self.fit_flag = False
        self.classes = np.arange(self.n_class)

        self.sampler = Sampler(oversampling_rate=self.config.oversampling_rate)
        self.no_optimization = NoOptimization(n_class=self.n_class)

        if self.config.optimization_method == 'soea':
            self.optimization = SOEAOptimization(soea_meta=config.soea_meta, n_class=self.n_class)
        elif self.config.optimization_method == 'moea':
            self.optimization = MOEAOptimization(moea_meta=config.moea_meta, n_class=self.n_class)
        elif self.config.optimization_method == 'grid':
            self.optimization = GridOptimization(grid_meta=config.grid_meta, n_class=self.n_class)
        elif self.config.optimization_method == 'no':
            self.optimization = NoOptimization(n_class=self.n_class)
        else:
            raise NotImplementedError()

    def update_buffer(self, datum, label):
        datum = datum.squeeze()
        label = label.item()
        self.data_buffer.append(datum)
        self.label_buffer.append(label)

        if len(self.data_buffer) > self.config.evaluation_meta.model_buffer_delay:
            self.data_buffer.pop(0)
            self.label_buffer.pop(0)

    def step_batch(self, data, labels, train_validate_split: float, priors: List[ndarray], evolving: bool = True,
                   max_generation: int = 10):
        n_instance, n_feature = data.shape
        assert 0 <= train_validate_split <= 1
        priors = np.array(priors)

        n_validate = int(n_instance * train_validate_split)
        train_data = data[:-n_validate]
        train_labels = labels[:-n_validate]
        validate_data = data[-n_validate:]
        validate_labels = labels[-n_validate:]
        # train_data, train_labels = self.sampler.sample(train_data, train_labels)
        self.fit_batch(train_data, train_labels)

        for datum, label in zip(data, labels):
            self.update_buffer(datum, label)
        self.cost_adapt(priors, validate_data, validate_labels, evolving=evolving, max_generation=max_generation)

    def step(self, datum: ndarray, label: ndarray, evolving: bool, priors: ndarray, fit_flag: bool = True):
        label = np.atleast_1d(label)
        datum = np.atleast_2d(datum)

        prob_uniform = self.predict_prob(datum)
        prob_naive = self.prob_naive(prob_uniform)
        prob_single = self.prob_single(prob_uniform)
        prob_ensemble = self.prob_ensemble(prob_uniform)

        predict_uniform = prob_uniform.argmax()
        predict_naive = prob_naive.argmax()
        predict_single = prob_single.argmax()
        predict_ensemble = prob_ensemble.argmax()

        if fit_flag:
            self.fit(datum, label)
        self.update_buffer(datum, label)

        self.cost_adapt(prior_vectors=priors, data=self.data_buffer, labels=self.label_buffer, evolving=evolving)

        vector_naive = self.no_optimization.vector_optimal()
        vector_single = self.optimization.vector_optimal()
        vector_ensemble = self.optimization.vector_ensemble()
        vectors_ensemble = self.optimization.vectors()
        return (
            prob_uniform, prob_naive, prob_single, prob_ensemble,
            predict_uniform, predict_naive, predict_single, predict_ensemble,
            vector_naive, vector_single, vector_ensemble, vectors_ensemble
        )

    def predict_prob(self, datum):
        if not self.fit_flag:
            prob = np.ones(shape=(1, self.n_class)) / self.n_class
            return prob
        else:
            if isinstance(self.classifier, HoeffdingTreeClassifier):
                if not isinstance(datum, ndarray):
                    datum = datum.numpy()
                return torch.from_numpy(self.classifier.predict_proba(datum) + 1e-16)
            else:

                return self.classifier.predict_proba(datum) + 1e-16

    def prob_naive(self, prob):
        return self.no_optimization.predict_single(prob)

    def prob_single(self, prob):
        return self.optimization.predict_single(prob)

    def prob_ensemble(self, prob):
        return self.optimization.predict_ensemble(prob)

    def fit(self, datum, label):
        self.classifier.partial_fit(torch.tensor(datum, dtype=torch.float32), torch.tensor(label, dtype=torch.long))
        self.fit_flag = True

    def fit_batch(self, data, labels):
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        self.classifier.fit(data, labels)
        self.fit_flag = True

    def cost_adapt(self, prior_vectors: ndarray, data, labels, evolving: bool, max_generation=None):
        if self.config.optimization_method == 'no':
            self.optimization.optimize(prior_vectors, None, None)
        elif evolving:
            data_resampled, labels_resampled = self.sampler.sample(data, labels)
            predicts_resampled = self.predict_prob(data_resampled)
            self.optimization.optimize(prior_vectors, labels_resampled, predicts_resampled,
                                       max_generation=max_generation)
        self.no_optimization.optimize(prior_vectors, None, None)


class Optimization:
    def vector_optimal(self):
        raise NotImplementedError()

    def vector_ensemble(self):
        raise NotImplementedError()

    def vectors(self):
        raise NotImplementedError()

    def predict_single(self, prob):
        raise NotImplementedError()

    def predict_ensemble(self, prob):
        raise NotImplementedError()

    def optimize(self, prior_vectors: ndarray, labels: ndarray, predicts: ndarray, **kwargs):
        raise NotImplementedError()

    def select_k_optimal(self, k: int, population):
        candidate = []
        for individual in population:
            candidate.append((0 if individual.FitnV is None else individual.FitnV, individual))
        candidate.sort(key=lambda x: x[0], reverse=True)

        return [self.config.decoder(individual[1].Chrom.flatten()) for individual in candidate[:k]]

    def select_k_optimal_individual(self, k: int, population):
        candidate = []
        for individual in population:
            candidate.append((0 if individual.ObjV is None else individual.ObjV, individual))
        candidate.sort(key=lambda x: self.config.problem_meta.select_key(x[0]), reverse=True)
        return [c[1] for c in candidate[:k]]


class NoOptimization(Optimization):
    def __init__(self, n_class):
        self.vector = np.ones(shape=(1, n_class)) / n_class

    def vector_optimal(self):
        return self.vector

    def vector_ensemble(self):
        return self.vector

    def vectors(self):
        return np.array([self.vector])

    def predict_single(self, prob):
        prob = prob * self.vector
        return prob / prob.sum()

    def predict_ensemble(self, prob):
        return self.predict_single(prob)

    def optimize(self, prior_vectors: ndarray, labels: ndarray, predicts: ndarray, **kwargs):
        self.vector = prior_vectors[0]


class GridOptimization(Optimization):
    def __init__(self, grid_meta: GridMeta, n_class: int):
        self.grid_meta = grid_meta
        self.n_class = n_class
        self.population = uniform_chromosome(n_class, grid_meta.n_individual)
        self._vector = np.ones(shape=(1, n_class)) / n_class
        self._vectors = [deepcopy(self._vector) for _ in range(grid_meta.n_individual)]
        self.fitness = Fitness(self.n_class, func=GmeanObjective())

    def vector_optimal(self):
        return self._vector

    def vector_ensemble(self):
        return np.mean(self._vectors, axis=0)

    def vectors(self):
        return self._vectors

    def predict_single(self, prob):
        vector = self.vector_optimal()
        prob = prob * vector
        return prob / prob.sum()

    def predict_ensemble(self, prob):
        vector = self.vector_ensemble()
        prob = prob * vector
        return prob / prob.sum()

    def optimize(self, prior_vectors: ndarray, labels, predicts, max_generation: int = None):
        prophet = np.vstack((prior_vectors, self.population))[:self.grid_meta.n_individual]
        vector_star = None
        max_fit = None
        for vector in prophet:
            fit = self.fitness.fitness(probs=predicts, labels=labels, vector=vector)
            if vector_star is None or fit > max_fit:
                vector_star = vector
                max_fit = fit

        self._vector = vector_star
        self._vectors = prophet
        self._vector = self._vector / self._vector.sum()
        self._vectors = [vector / vector.sum() for vector in self._vectors]


class SOEAOptimization(Optimization):
    def __init__(self, soea_meta: SOEAMeta, n_class: int):
        self.soea_meta = soea_meta
        self.n_class = n_class
        self.population = ea.Population(Encoding='RI', NIND=soea_meta.n_individual,
                                        Chrom=uniform_chromosome(n_class, soea_meta.n_individual))
        self._vector = np.ones(shape=(1, n_class)) / n_class
        self._vectors = [deepcopy(self._vector) for _ in range(soea_meta.n_individual)]
        self.log = False
        self.prior_population = None
        self.std = 0

    def vector_optimal(self):
        return self._vector

    def vector_ensemble(self):
        return np.mean(self._vectors, axis=0)

    def vectors(self):
        return self._vectors

    def predict_single(self, prob):
        vector = self.vector_optimal()
        prob = prob * vector
        return prob / prob.sum()

    def predict_ensemble(self, prob):
        vector = self.vector_ensemble()
        prob = prob * vector
        return prob / prob.sum()

    def optimize(self, prior_vectors: ndarray, labels, predicts, max_generation: int = None):
        prior_vectors = np.array(prior_vectors)
        self.prior_population = prior_vectors
        if max_generation is None:
            max_generation = self.soea_meta.max_generation
        algorithm = self.soea_meta.ea_algorithm(
            SOEACostVectorProblem(labels=labels, predicts=predicts, n_class=self.n_class),
            population=self.population, MAXGEN=max_generation,
            logTras=self.log, trappedValue=None, maxTrappedCount=None)

        # assert self.config.ea_meta.n_prophet_plugin <= self.config.ea_meta.n_individual

        solution = ea.optimize(algorithm, seed=0, verbose=self.log, prophet=prior_vectors,
                               saveFlag=False, drawing=0, outputMsg=self.log, drawLog=False)
        last_pop = solution['lastPop']
        opt_pop = solution['optPop']
        self.population = last_pop
        self._vector = opt_pop.Chrom.squeeze()
        self._vectors = [pop.Chrom.squeeze() for pop in last_pop]
        self._vector = self._vector / self._vector.sum()
        self._vectors = [vector / vector.sum() for vector in self._vectors]
        self.std = np.std(last_pop.ObjV)


class MOEAOptimization(Optimization):
    def __init__(self, moea_meta: MOEAMeta, n_class: int):
        self.moea_meta = moea_meta
        self.n_class = n_class
        self.population = ea.Population(Encoding='RI', NIND=moea_meta.n_individual,
                                        Chrom=uniform_chromosome(n_class, moea_meta.n_individual))
        self.vector = np.ones(shape=(1, n_class)) / n_class
        self.vectors = [deepcopy(self.vector) for _ in range(moea_meta.n_individual)]

    def vector_optimal(self):
        return self.vector

    def vector_ensemble(self):
        return np.mean(self.vectors, axis=0)

    def vectors(self):
        return self.vectors

    def predict_single(self, prob):
        vector = self.vector_optimal()
        prob = prob * vector
        return prob / prob.sum()

    def predict_ensemble(self, prob):
        vector = self.vector_ensemble()
        prob = prob * vector
        return prob / prob.sum()

    def vector_moea_algorithm(self, prior_vectors: ndarray, labels, predicts, population, max_generation):
        # TODO:
        ...


class Sampler:
    def __init__(self, retain_size=-1, oversampling_rate=1, k_neighbors=5):
        self.retain_size = retain_size
        self.oversampling_rate = oversampling_rate
        self.k_neighbors = k_neighbors

    def sample(self, X, y):
        """
        Apply SMOTE to oversample the imbalance_dataset.

        Parameters:
        - X: 2D array-like or pandas DataFrame, shape (n_samples, n_features)
            The input samples.
        - y: 1D array-like or pandas Series, shape (n_samples,)
            The class labels.
        - oversampling_rate: int, optional (default=1)
            The rate of oversampling. For example, if oversampling_rate=2, the imbalance_dataset size will be doubled.
        - k_neighbors: int, optional (default=5)
            The number of nearest neighbors to be used in the SMOTE algorithm.

        Returns:
        - X_resampled: 2D ndarray, shape (new_samples, n_features)
            The resampled imbalance_dataset.
        - y_resampled: 1D ndarray, shape (new_samples,)
            The corresponding class labels for the resampled imbalance_dataset.
        """
        X = np.array(X)
        y = np.array(y)
        if self.oversampling_rate != 1:
            # Initialize a Nearest Neighbors classifier
            knn = NearestNeighbors(n_neighbors=self.k_neighbors)

            # Fit the classifier on the original data
            knn.fit(X)

            # Lists to store the resampled data
            X_resampled = []
            y_resampled = []

            # Loop over each sample in the original data
            size = int(self.oversampling_rate - 1)

            for i in range(len(X)):
                # Find k-nearest neighbors of the current sample
                indices = knn.kneighbors([X[i]], return_distance=False)
                # Randomly choose one of the nearest neighbors
                chosen_neighbor_indices = np.random.choice(indices[0, 1:], size=size)

                for index in chosen_neighbor_indices:
                    # Generate a synthetic sample as a convex combination of the chosen sample and neighbor
                    synthetic_sample = X[i] + np.random.rand() * (X[index] - X[i])

                    # Append the synthetic sample and its label to the resampled data
                    X_resampled.append(synthetic_sample)
                    y_resampled.append(y[i])

            # Concatenate the original data with the resampled data
            X = np.vstack((X, np.array(X_resampled)))
            y = np.concatenate((y, np.array(y_resampled)))

        if self.retain_size == -1:
            return X, y

        indexes = np.arange(len(X))
        np.random.shuffle(indexes)
        indexes = indexes[:self.retain_size]
        return X[indexes], y[indexes]
