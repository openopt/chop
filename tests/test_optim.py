"""Tests full gradient batch-wise optimization algorithms from chop.optim"""


import torch
from chop import optim
from chop import utils
from chop import constraints
from chop.utils import logging

import pytest

# Set up a batch of toy constrained optimization problems
batch_size = 20
d = 2
xstar = torch.rand(batch_size, d)
alpha = .5
constraint = constraints.LinfBall(alpha)
xstar = constraint.prox(xstar)

# Minimize quadratics for each datapoint in the batch
@utils.closure
def loss_fun(x):
    return .5 * ((x - xstar) ** 2).view(batch_size, -1).sum(-1)


@pytest.mark.parametrize('step', [1., 'backtracking'])
def test_minimize_pgd(step):
    max_iter = 2000
    x0 = torch.zeros_like(xstar)
    trace_cb = logging.Trace(closure=loss_fun)

    sol = optim.minimize_pgd(loss_fun, x0, constraint.prox,
                             step=step,
                             max_iter=max_iter, callback=trace_cb)

    assert sol.certificate.allclose(torch.zeros(batch_size, dtype=torch.float)), sol.certificate


def test_minimize_frank_wolfe():
    max_iter = 2000
    x0 = torch.zeros_like(xstar)
    sol = optim.minimize_frank_wolfe(loss_fun, x0, constraint.lmo,
                                     max_iter=max_iter)
    assert sol.certificate.allclose(torch.zeros(batch_size, dtype=torch.float), atol=1e-3), sol.certificate


def test_minimize_three_split():
    max_iter = 200
    x0 = torch.zeros_like(xstar)
    batch_size = x0.size(0)
    sol = optim.minimize_three_split(loss_fun, x0, constraint.prox,
                                     max_iter=max_iter)

    cert = sol.certificate
    assert cert.allclose(torch.zeros(batch_size, dtype=torch.float), atol=1e-5)