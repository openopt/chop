"""Tests for utility functions"""

from constopt.utils import closure
import torch
from torch import nn
from constopt import utils


# Set up random regression problem
alpha = 1.
n_samples, n_features = 20, 15
X = torch.rand((n_samples, n_features))
w = torch.rand(n_features)
w = alpha * w / sum(abs(w))
y = X.mv(w)
# Logistic regression: \|y\|_\infty <= 1
y = abs(y / y.max())

tol = 4e-3


# def test_init_lipschitz():
criterion = nn.MSELoss(reduction='none')

@closure
def loss_fun(X):
    return criterion(X.mv(w), y)


L = utils.init_lipschitz(loss_fun, X.detach().clone().requires_grad_(True))
print(L)
