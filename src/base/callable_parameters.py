from abc import abstractmethod

from numpy import ndarray

from src.base.config_store import JSONAble


class CallableParameter(JSONAble):
    def to_dict(self):
        return self.__class__.__name__

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())


##########################
#   objective functions  #
# single obj/ multi objs #
##########################

class ObjectiveFunction(CallableParameter):
    @abstractmethod
    def __call__(self, accuracy, gmean, precision, recall, fscore, support, confusion):
        """
        calculate objective by metric
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def m(self) -> int:
        """
        give the number of objective
        """
        raise NotImplementedError()

    @m.setter
    def m(self, v):
        ...


class GmeanObjective(ObjectiveFunction):
    """
    gm
    """

    def m(self):
        return 1

    def __call__(self, accuracy, gmean, precision, recall, fscore, support, confusion):
        return gmean


##################################################
#   constraint functions (two / multi classes)   #
##################################################
class ConstraintFunction(CallableParameter):
    @abstractmethod
    def __call__(self, matrix, prior):
        raise NotImplementedError()


class NoConstraint(ConstraintFunction):
    """
    no constraint
    """

    def __call__(self, matrix, prior):
        return 0


class SelectionFunction(CallableParameter):
    @abstractmethod
    def __call__(self, objVs: ndarray):
        raise NotImplementedError()


class SF1(SelectionFunction):
    def __call__(self, objVs: ndarray):
        return objVs.prod() ** (1 / objVs.size)


if __name__ == '__main__':
    ...
