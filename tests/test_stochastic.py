"""Tests for constrained optimizers"""

import numpy as np
import torch
from torch.autograd import Variable
import pytest
import shutil
from cox.store import Store

import chop
from chop import stochastic


OUT_DIR = "logging/tests/test_optim"
shutil.rmtree(OUT_DIR, ignore_errors=True)
MAX_ITER = 300

torch.manual_seed(0)

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


@pytest.mark.parametrize('algorithm', [stochastic.PGD, stochastic.PGDMadry,
                                       stochastic.FrankWolfe])
@pytest.mark.parametrize('lr', [1., .5, .1, .05, .001])
def test_L1Ball(algorithm, lr):
    # Setup
    constraint = chop.constraints.L1Ball(alpha)
    assert (constraint.prox(w) == w).all()
    w_t = Variable(torch.zeros_like(w), requires_grad=True)

    optimizer = algorithm([w_t], constraint, lr=lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    # Logging
    store = Store(OUT_DIR)
    store.add_table('metadata', {'algorithm': str, 'lr': float})

    store['metadata'].append_row({'algorithm': optimizer.name, 'lr': lr})
    store.add_table(optimizer.name, {'func_val': float, 'certificate': float,
                                     'norm(w_t)': float})
    cert = torch.tensor(np.inf)
    for ii in range(MAX_ITER):
        optimizer.zero_grad()
        loss = criterion(X.mv(w_t), y)
        loss.backward()

        optimizer.step()

        cert = next(optimizer.certificate)  # only one parameter here

        store.log_table_and_tb(optimizer.name, {'func_val': loss.item(),
                                                'certificate': cert.item(),
                                                'norm(w_t)': sum(abs(w_t)).item()
                                                })
        store[optimizer.name].flush_row()

    store.close()
