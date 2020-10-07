"""Tests for constrained optimizers"""

import numpy as np
import torch
from torch.autograd import Variable
import pytest
import shutil
from cox.store import Store

import constopt
from constopt import optim


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


@pytest.mark.parametrize('algorithm', [optim.PGD, optim.PGDMadry,
                                       optim.FrankWolfe, optim.MomentumFrankWolfe])
@pytest.mark.parametrize('step_size', [1., .5, .1, .05, .001, 0.])
def test_L1Ball(algorithm, step_size):
    # Setup
    constraint = constopt.constraints.L1Ball(alpha)
    assert (constraint.prox(w) == w).all()
    w_t = Variable(torch.zeros_like(w), requires_grad=True)

    optimizer = algorithm([w_t], constraint)
    criterion = torch.nn.MSELoss(reduction='mean')

    # Logging
    store = Store(OUT_DIR)
    store.add_table('metadata', {'algorithm': str, 'step-size': float})

    store['metadata'].append_row({'algorithm': optimizer.name, 'step-size': step_size})
    store.add_table(optimizer.name, {'func_val': float, 'FW gap': float,
                                     'norm(w_t)': float})
    gap = torch.tensor(np.inf)
    for ii in range(MAX_ITER):
        optimizer.zero_grad()
        loss = criterion(X.mv(w_t), y)
        loss.backward()

        # Compute gap
        with torch.no_grad():
            gap = constraint.fw_gap(w_t.grad, w_t)

        optimizer.step(step_size)
        store.log_table_and_tb(optimizer.name, {'func_val': loss.item(),
                                                'FW gap': gap.item(),
                                                'norm(w_t)': sum(abs(w_t)).item()
                                                })
        store[optimizer.name].flush_row()

    store.close()
