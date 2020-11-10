"""Tests for utility functions"""

import torch
from torch import nn
from constopt import opt_utils


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


def test_init_lipschitz():
    criterion = nn.CrossEntropyLoss()

    def loss_fun(w):
        return criterion(X.mv(w), y)

    w0 = torch.rand(n_features)

    f0 = loss_fun(w0)
    f0.backward()

    L = opt_utils.init_lipschitz(loss_fun, w0.grad, w0)
