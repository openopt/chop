"""
L2-penalized Logistic Regression (full-batch)
==================================

L2 penalized (unconstrained) logistic regression on the Covtype dataset.
Uses full-batch gradient descent with line-search.
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import chop

from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Regularization strength
lmbd = 1.

max_iter = 200

# Load and prepare dataset
X, y = fetch_covtype(return_X_y=True)
y[y != 2] = -1
y[y == 2] = 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.float32, device=device)

n_datapoints, n_features = X.shape

# Initialize weights
x0 = torch.zeros(1, n_features, dtype=X.dtype, device=X.device)

# Binary cross entropy
@chop.utils.closure
def logloss_reg(x, pen=lmbd):
    y_X_x = y * (X @ x.flatten())
    l2 = 0.5 * x.pow(2).sum()
    logloss = torch.log1p(torch.exp(-y_X_x)).sum()
    return (logloss + pen * l2) / X.size(0)


@torch.no_grad()
def log_accuracy(kwargs):
    out = X @ kwargs['x'].flatten()
    acc = (torch.sign(out).detach().cpu().numpy().round() == y.cpu().numpy()).mean()
    return acc


callback = chop.utils.logging.Trace(callable=lambda kwargs: (log_accuracy(kwargs),
                                                             logloss_reg(kwargs['x'], pen=0.,
                                                                         return_jac=False).item()))

result = chop.optim.minimize_pgd(logloss_reg, x0, callback=callback, max_iter=max_iter, step='backtracking')

fig = plt.figure()
plt.plot(np.array([val.item() for val in callback.trace_f]).clip(0, 1))
plt.title("Regularized Loss")
plt.show()

accuracies, losses = zip(*callback.trace_callable)

for name, vals in (('Accuracy', accuracies),
                   ('Loss', losses)):
    fig = plt.figure()
    plt.title(name)
    plt.plot(vals)
    plt.show()
