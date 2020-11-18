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


def certificate_prox(kwargs):
    x = kwargs['x']
    prox = kwargs['prox']
    grad = kwargs['grad']
    return torch.norm((x - prox(x - grad, 1.)).view(x.size(0), -1), dim=-1)


def test_minimize_pgd():
    max_iter = 100
    x0 = torch.zeros_like(xstar)
    trace_cb = logging.Trace(closure=loss_fun, callable=certificate_prox)

    sol = optim.minimize_pgd(loss_fun, x0, constraint.prox,
                             step=1.,
                             max_iter=max_iter, callback=trace_cb)
    x = sol.x
    grad = sol.grad
    prox = constraint.prox
    cert = certificate_prox(locals())
    assert cert.allclose(torch.zeros(batch_size, dtype=torch.float), atol=1e-3), cert


def test_minimize_frank_wolfe():
    max_iter = 1000
    x0 = torch.zeros_like(xstar)
    sol = optim.minimize_frank_wolfe(loss_fun, x0, constraint.lmo,
                                     max_iter=max_iter)
    x = sol.x
    grad = sol.grad
    cert = constraint.fw_gap(grad, x)
    assert cert.allclose(torch.zeros(batch_size, dtype=torch.float), atol=1e-4), cert


def test_minimize_three_split():
    max_iter = 200
    x0 = torch.zeros_like(xstar)
    batch_size = x0.size(0)
    sol = optim.minimize_three_split(loss_fun, x0, constraint.prox,
                                     max_iter=max_iter)

    cert = sol.certificate
    assert cert.allclose(torch.zeros(batch_size, dtype=torch.float), atol=1e-5)