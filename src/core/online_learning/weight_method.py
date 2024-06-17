import torch
from torch import Tensor


def weighted_predict_cost_vector(prob: Tensor, vector: Tensor, batch: bool):
    assert isinstance(prob, Tensor) and isinstance(vector, Tensor)
    prob = torch.atleast_2d(prob)
    vector = torch.atleast_2d(vector)
    weighted_prob = prob * vector
    weighted_prob = weighted_prob / weighted_prob.sum(dim=1, keepdims=True)
    if batch:
        return weighted_prob
    else:
        return weighted_prob.squeeze()


class CostVector:
    def __init__(self, vector, batch: bool = True):
        self.vector = vector
        self.batch = batch

    def __call__(self, prob):
        return weighted_predict_cost_vector(prob=torch.tensor(prob),
                                            vector=torch.tensor(self.vector),
                                            batch=self.batch)


if __name__ == '__main__':
    ...
