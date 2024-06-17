from __future__ import annotations

import itertools
import json
import os
import pickle
from typing import List, Generator, Literal, Type

import geatpy as ea
import numpy as np

from src.core.evolution.algorithms import DE_best_1_L

np.float = float
import pandas as pd
import torch
from skmultiflow.trees import HoeffdingTreeClassifier
from torch import nn

from src.base.callable_parameters import SelectionFunction, ObjectiveFunction, ConstraintFunction, NoConstraint, \
    GmeanObjective, SF1
from src.base.config_store import JSONAble
from src.helper.helper import chromosome_to_phenotype

current_path = os.path.dirname(__file__)


def get_device():
    return 'cpu'


torch.set_default_device(get_device())


def null_or_default(nullable, default):
    return default if nullable is None else nullable


def to_dict(obj, exclude: List = frozenset()):
    object_dict = {}
    for key in dir(obj):
        if key.startswith('__') or key.startswith('_'):
            continue
        if key in exclude:
            continue
        attr = getattr(obj, key)

        if isinstance(attr, JSONAble):
            object_dict[key] = attr.to_dict()
            continue

        if isinstance(attr, type):
            object_dict[key] = attr.__name__
        if callable(attr):
            continue
        if isinstance(attr, np.ndarray):
            object_dict[key] = attr.tolist()
            continue
        object_dict[key] = attr

    return object_dict


class Configuration(JSONAble):
    device = get_device()

    def __init__(self, parameter: ComparisonParameter, debug: bool = False,
                 comparison_mode: Literal['ir', 'overall_ir', 'grid'] = 'ir', pretrain_ratio: float = 0.3,
                 train_validate_split: float = 0.5, oversampling_rate: int = 3,
                 init_mode: Literal['disturbance', 'exact', 'grid', 'random'] = 'disturbance'):
        self.comparison_mode = comparison_mode
        self.optimization_method = parameter.optimization_method
        self.oversampling_rate = oversampling_rate
        self.init_mode = init_mode

        self.pretrain_ratio = pretrain_ratio
        self.train_validate_split = train_validate_split

        self.parameter = parameter
        self.debug = debug
        self.plot = False

        self.soea_meta = SOEAMeta()
        self.moea_meta = MOEAMeta()
        self.gd_meta = GDMeta()
        self.bo_meta = BOMeta()
        self.grid_meta = GridMeta()
        self.dataset_meta = DatasetMeta(dataset=parameter.dataset)
        self.problem_meta = ProblemMeta(dim=self.dataset_meta.n_class * (self.dataset_meta.n_class - 1))
        self.online_learning_meta = OnlineLearningMeta(
            ensemble_size=parameter.ensemble_size,
            classifier=parameter.classifier
        )
        self.evaluation_meta = EvaluationMeta(
            model_buffer_delay=parameter.model_buffer_delay,
            validate_size=parameter.model_buffer_delay
        )
        self.seed = parameter.seed

        self.decoder = self.chrom_to_vector

    def chrom_to_vector(self, chrom):
        return chrom

    def chrom_to_matrix(self, chrom):
        return chromosome_to_phenotype(self.dataset_meta.n_class, chrom, self.soea_meta.flexible_diagonal)

    def to_path(self, base_path: str = None):
        if base_path is None:
            base_path = f'{current_path}/../result'
            # f'{current_path}/../result_tmp'
        return self.get_parameters().to_path(base_path)

    def get_parameters(self) -> ComparisonParameter:
        return self.parameter

    def get(self, obj_name: Literal['predict', 'target', 'matrix', 'time', 'objective', 'output_n']):
        return np.load(f'{self.to_path()}/{obj_name}.npy')

    def to_dict(self):
        return to_dict(self)

    def log(self, message=''):
        if self.debug:
            print(message)


class SOEAMeta(JSONAble):
    def __init__(
            self,
            search_bound: float = np.inf,
            max_generation: int = 10,
            n_individual: int = 100,
            n_prophet_plugin: int = 50,
            algorithm: Type[ea.Algorithm] = DE_best_1_L,
            flexible_diagonal: bool = False,
            trapped_value: float = None,
            debug: bool = False
    ):
        self.search_bound = search_bound
        self.max_generation = max_generation
        self.n_individual = n_individual
        self.n_prophet_plugin = null_or_default(n_prophet_plugin, self.n_individual // 2)
        self.flexible_diagonal = flexible_diagonal
        self.ea_algorithm = algorithm
        self.trapped_value = trapped_value
        self.debug = debug

    def to_dict(self):
        return to_dict(self)


class MOEAMeta(JSONAble):
    def __init__(
            self,
            search_bound: float = np.inf,
            max_generation: int = 10,
            n_individual: int = 100,
            n_prophet_plugin: int = 50,
            ea_algorithm: Type[ea.Algorithm] = ea.moea_MOEAD_DE_templet,
            flexible_diagonal: bool = False,
            trapped_value: float = None,
            debug: bool = False
    ):
        self.search_bound = search_bound
        self.max_generation = max_generation
        self.n_individual = n_individual
        self.n_prophet_plugin = null_or_default(n_prophet_plugin, self.n_individual // 2)
        self.flexible_diagonal = flexible_diagonal
        self.ea_algorithm = ea_algorithm
        self.trapped_value = trapped_value
        self.debug = debug

    def to_dict(self):
        return to_dict(self)


class BOMeta(JSONAble):
    def __init__(self, n_iter: int = 30, n_init_points: int = 10, debug: int = 0):
        self.n_iter = n_iter
        self.n_init_points = n_init_points
        self.debug = debug

    def to_dict(self):
        return to_dict(self)


class GDMeta(JSONAble):
    def __init__(self, n_iter: int = 50, decay: float = 1, debug: bool = False):
        self.n_iter = n_iter
        self.decay = decay
        self.debug = debug

    def to_dict(self):
        return to_dict(self)


class GridMeta(JSONAble):
    def __init__(
            self,
            n_individual: int = 100,
            n_prophet_plugin: int = 50,
            flexible_diagonal: bool = False
    ):
        self.n_individual = n_individual
        self.n_prophet_plugin = null_or_default(n_prophet_plugin, self.n_individual // 2)
        self.flexible_diagonal = flexible_diagonal

    def to_dict(self):
        return to_dict(self)


class OnlineLearningMeta(JSONAble):
    def __init__(self,
                 online_classifier_lr: float = 0.005,
                 evaluation_classifier_lr: float = 0.1,
                 ensemble_size: int = 5,
                 fading_factor: float = 0.99,
                 criterion=nn.CrossEntropyLoss(),
                 classifier=HoeffdingTreeClassifier):
        self.criterion = criterion
        self.ensemble_size = ensemble_size
        self.online_classifier_lr = online_classifier_lr
        self.evaluation_classifier_lr = evaluation_classifier_lr
        self.classifier = classifier
        self.fading_factor = fading_factor

    def to_dict(self):
        return to_dict(self)


class EvaluationMeta(JSONAble):
    def __init__(self,
                 evolution_interval: int = 10,
                 model_buffer_delay: int = 600,
                 validate_size: int = 300,
                 eval_metric: Literal[
                     'offline',
                     'overall',
                     'latest'
                 ] = 'offline'):
        self.model_buffer_delay = model_buffer_delay
        self.evolution_interval = evolution_interval
        self.eval_metric = eval_metric
        self.validate_size = validate_size

    def to_dict(self):
        return to_dict(self)


class ProblemMeta(JSONAble):
    def __init__(self, dim: int,
                 name: str = 'CostMatrixProblem',
                 M: int = 1,
                 search_bound: int = 50,
                 maxormins: List[Literal[1, -1]] = None,
                 varTypes: List[Literal[0, 1]] = None,
                 constraint_function: ConstraintFunction = NoConstraint(),
                 objective_function: ObjectiveFunction = GmeanObjective(),
                 selection_function: SelectionFunction = SF1(),
                 lb: List = None,
                 ub: List = None):
        self.M = M
        self.maxormins = null_or_default(maxormins, [-1] * self.M)
        self.constraint_function = constraint_function
        objective_function.m = M
        self.objective_function = objective_function
        self.select_key = selection_function
        self.search_bound = search_bound
        self.name = name
        self.dim = dim
        self.varTypes = null_or_default(varTypes, [0] * self.dim)
        self.lb = null_or_default(lb, [0] * self.dim)
        self.ub = null_or_default(ub, [self.search_bound] * dim)

    def to_dict(self):
        return to_dict(self)


class DatasetMeta(JSONAble):
    dataset_base_path: str = f'{current_path}/../../imbalance_dataset'

    def __init__(self,
                 dataset: str,
                 shuffle: bool = False
                 ):
        self.dataset = dataset
        self.shuffle = shuffle
        self._data = None
        self._labels = None

        overview = self.load_overview(self.dataset)

        self.n_class = overview['n_class']
        self.n_instance = overview['n_instance']
        self.n_feature = overview['n_feature']
        self.n_instance_per_class = overview['n_instance_per_class']
        self.imbalance_ratio = overview['imbalance_ratio']
        ipc = dict(sorted(self.n_instance_per_class.items()))
        self.label_set = list(ipc.keys())
        self.class_sizes = np.array(list(ipc.values()))

    @property
    def data(self):
        if self._data is None:
            self._load_data()
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            self._load_data()
        return self._labels

    def _load_data(self):
        read_data = pd.read_csv(f'{self.dataset_base_path}/{self.dataset}.csv', header=None)
        if self.shuffle:
            read_data.sample(frac=1)
        read_data = np.array(read_data)
        self._labels = read_data[:, -1].astype(int)
        self._data = read_data[:, :-1]

    def to_dict(self):
        return to_dict(self, ['_data', '_labels', 'data', 'labels'])

    def load_overview(self, dataset):
        with open(f'{current_path}/../base/dataset_overview.json', 'r') as file:
            dataset_dict = json.load(file)
        return dataset_dict[dataset]


class ComparisonParameter(JSONAble):
    def __init__(self,
                 ensemble_size: int,
                 buffer_size: int,
                 classifier,
                 dataset: str,
                 seed: int,
                 optimization_method: Literal['soea', 'moea', 'bo', 'gd', 'enum'],
                 ):
        self.dataset = dataset
        self.ensemble_size = ensemble_size
        self.seed = seed
        self.classifier = classifier
        self.model_buffer_delay = buffer_size
        self.optimization_method = optimization_method

    def to_path(self, base_path: str = f'{current_path}/../result'):
        return f'{base_path}/' \
               f'{self.ensemble_size}/' \
               f'{self.model_buffer_delay}/' \
               f'{self.optimization_method}/' \
               f'{self.dataset}/' \
               f'{self.seed}'

    def get(self, obj_name: Literal['predict', 'target', 'matrix', 'time', 'objective']):
        return np.load(f'{self.to_path()}/{obj_name}.npy')

    def get_all(self, obj_name_list: List[Literal['predict', 'target', 'weight', 'time', 'objective', 'output']]):
        with open(f'{current_path}/results/result.pickle', 'rb') as f:
            results = pickle.load(f)
        return [results[obj_name] for obj_name in obj_name_list]

    def get_pickle(self, base_path: str = f'{current_path}/../result', filename: str = 'result'):
        with open(f'{self.to_path(base_path)}/{filename}.pickle', 'rb') as f:
            return pickle.load(f)

    def to_dict(self):
        return to_dict(self)

    def __getitem__(self, item):
        return to_dict(self)[item]

    def to_str(self, comparative_parameters: List[str]):
        strs = []
        for p in comparative_parameters:
            strs.append(f'{p}[{self[p]}]')
        return ','.join(strs)


if __name__ == '__main__':
    ...
