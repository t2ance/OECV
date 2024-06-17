import geatpy as ea
import numpy as np
from numpy import ndarray
from torch import Tensor

from src.base.callable_parameters import ConstraintFunction, GmeanObjective, NoConstraint
from src.base.configuration import Configuration
from src.base.evaluation import fitness, Fitness, FitnessGmean
from src.core.online_learning.weight_method import CostVector


class SOEACostVectorProblem(ea.Problem):
    def __init__(self, labels, predicts, n_class, constraint_function: ConstraintFunction = NoConstraint(),
                 objective_function=GmeanObjective(), refer: ndarray = None):
        self.labels = labels
        self.probs = predicts
        self.refer = refer
        self.n_class = n_class
        self.fitness = FitnessGmean(n_class)
        self.constraint_function = constraint_function
        self.objective_function = objective_function
        M = 1
        maxormins = [-1]

        varTypes = [0] * n_class
        lb = [0] * n_class
        ub = [50] * n_class
        ea.Problem.__init__(self, 'SOEACostVectorProblem', M, maxormins, n_class, varTypes, lb, ub)

    def evalVars(self, variables):
        evaluations = []
        constraints_offense = []
        for vector in variables:
            fitness = self.fitness.fitness(self.probs, self.labels, vector)
            evaluations.append(fitness)
            constraints_offense.append(self.constraint_function(vector, self.refer))

        ObjV = np.array(evaluations).reshape(-1, self.M)
        CV = np.array(constraints_offense).reshape(-1, 1)
        return ObjV, CV
