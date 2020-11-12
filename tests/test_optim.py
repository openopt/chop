"""Tests full gradient batch-wise optimization algorithms from constopt.optim"""


import torch
from constopt import optim
from constopt import utils
from constopt import constraints


# Set up a batch of toy constrained optimization problems
batch_size = 20
d = 2
xstar = torch.rand(batch_size, d)

# Minimize quadratics for each datapoint in the batch
@utils.closure
def loss_fun(x):
    return ((x - xstar) ** 2).view(batch_size, -1).sum(-1)


alpha = .5
constraint = constraints.LinfBall(alpha)


def certificate(x, grad, prox):
    return torch.norm((x - prox(x - grad, 1.)).view(x.size(0), -1), dim=-1)


def test_minimize_pgd():
    max_iter = 10
    step_size = torch.ones(batch_size, dtype=float) * 2. * alpha / max_iter
    x0 = torch.zeros_like(xstar)
    sol = optim.minimize_pgd(loss_fun, x0, constraint.prox,
                             step_size)

    cert = certificate(sol.x, sol.grad, constraint.prox)
    assert cert.allclose(torch.zeros(batch_size, dtype=torch.double)), cert