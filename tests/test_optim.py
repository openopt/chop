"""Tests full gradient batch-wise optimization algorithms from chop.optim"""


import torch
from chop import optim
from chop import utils
from chop import constraints
from chop import logging


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
xstar = constraint.prox(xstar)


def certificate(kwargs):
    x = kwargs['x']
    prox = kwargs['prox']
    grad = kwargs['grad']
    return torch.norm((x - prox(x - grad, 1.)).view(x.size(0), -1), dim=-1)


trace_cb = logging.Trace(closure=loss_fun, callable=certificate)


def test_minimize_pgd():
    max_iter = 100
    x0 = torch.zeros_like(xstar)
    sol = optim.minimize_pgd(loss_fun, x0, constraint.prox,
                             step=1.,
                             max_iter=max_iter, callback=trace_cb)
    x = sol.x
    grad = sol.grad
    prox = constraint.prox
    cert = certificate(locals())
    assert cert.allclose(torch.zeros(batch_size, dtype=torch.float)), cert
