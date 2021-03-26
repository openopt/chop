"""
L2 penalized logistic regression
==================================

L2 penalized (unconstrained) logistic regression on the Covtype dataset.
Uses full-batch gradient descent with line-search.
"""

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
y = torch.tensor(y > 0, dtype=torch.float32, device=device)

n_datapoints, n_features = X.shape

# Initialize weights
x0 = torch.zeros(1, n_features, dtype=X.dtype, device=X.device)

# Binary cross entropy
criterion = torch.nn.BCEWithLogitsLoss()


@chop.utils.closure
def logloss(x, pen=lmbd):
    alpha = pen / n_datapoints
    out = chop.utils.bmv(X, x)
    loss = criterion(out, y)
    reg = .5 * alpha * x.pow(2).sum()
    return loss + reg


@torch.no_grad()
def log_accuracy(kwargs):
    out = chop.utils.bmv(X, kwargs['x'])
    out = torch.sigmoid(out)
    acc = (out.detach().cpu().numpy().round() == y.cpu().numpy()).mean()
    return acc


callback = chop.utils.logging.Trace(callable=lambda kwargs: (log_accuracy(kwargs),
                                                             logloss(kwargs['x'], pen=0.,
                                                                     return_jac=False)))

result = chop.optim.minimize_pgd(logloss, x0, callback=callback, max_iter=max_iter)

fig = plt.figure()
plt.plot([val.item() for val in callback.trace_f])
plt.title("Regularized Loss")
plt.show()

accuracies, losses = zip(*callback.trace_callable)

for name, vals in (('Accuracy', accuracies),
                   ('Loss', losses)):
    fig = plt.figure()
    plt.title(name)
    plt.plot(vals)
    plt.show
