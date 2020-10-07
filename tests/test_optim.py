"""Tests for constrained optimizers"""

import numpy as np
import torch
from torch.autograd import Variable
import pytest
import constopt

from cox.store import Store

OUT_DIR = "logging/tests/"
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


@pytest.mark.parametrize('algorithm', ['vanilla-FW', 'momentum-FW', 'PGD', 'PGD-Madry'])
@pytest.mark.parametrize('step_size', [.1, .05, .001, 0.])
def test_L1Ball(algorithm, step_size):
    store = Store(OUT_DIR)
    store.add_table('metadata', {'algorithm': str, 'step-size': float})

    store['metadata'].append_row({'algorithm': algorithm, 'step-size': step_size})
    store.add_table(algorithm, {'func_val': float, 'FW gap': float,
                                'norm(w_t)': float})

    constraint = constopt.constraints.L1Ball(alpha)
    assert (constraint.prox(w) == w).all()
    w_t = Variable(torch.zeros_like(w), requires_grad=True)

    if algorithm == "vanilla-FW":
        optimizer = constopt.optim.FrankWolfe([w_t], constraint.lmo)
    elif algorithm == 'momentum-FW':
        optimizer = constopt.optim.MomentumFrankWolfe([w_t], constraint.lmo)
    elif algorithm == 'PGD':
        optimizer = constopt.optim.PGD([w_t], constraint.prox)
    elif algorithm == 'PGD-Madry':
        optimizer = constopt.optim.PGDMadry([w_t], constraint.prox,
                                            constraint.lmo)
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(X.mv(w_t), y)
    ii = 0
    gap = torch.tensor(np.inf)
    for ii in range(MAX_ITER):
        store.log_table_and_tb(algorithm, {'func_val': loss.item(),
                                           'FW gap': gap.item(),
                                           'norm(w_t)': sum(abs(w_t)).item()
                                           })
        store[algorithm].flush_row()
        loss.backward()
        with torch.no_grad():
            gap = constraint.fw_gap(w_t.grad, w_t)
        optimizer.step(step_size)
        loss = criterion(X.mv(w_t), y)

        ii += 1
    store.close()
