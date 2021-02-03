"""
Stochastic constrained optimization dynamics.
================================================
Sets up simple 2-d problems on Linf balls to visualize dynamics of various
stochastic constrained optimization algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from chop.constraints import LinfBall
from chop.stochastic import PGD, PGDMadry, FrankWolfe

torch.random.manual_seed(0)

OPTIMIZER_CLASSES = [PGD, PGDMadry, FrankWolfe]


def setup_problem(make_nonconvex=False):
    radius = 1.
    x_star = torch.tensor([radius, radius/2])
    x_0 = torch.zeros_like(x_star)

    def loss_func(x):
        val = .5 * ((x - x_star) ** 2).sum()
        if make_nonconvex:
            val += .1 * torch.sin(50 * torch.norm(x, p=1) + .1)
        return val

    constraint = LinfBall(radius)

    return x_0, x_star, loss_func, constraint


def optimize(x_0, loss_func, constraint, optimizer_class, iterations=10):
    x = x_0.detach().clone()
    x.requires_grad = True
    # Use Madry's heuristic for step size
    lr = {
        FrankWolfe: 2.5 / iterations,
        PGD: 2.5 * constraint.alpha / iterations * 2.,
        PGDMadry: 2.5 / iterations
    }
    optimizer = optimizer_class([x], constraint, lr=lr[optimizer_class])
    iterates = [x.data.numpy().copy()]
    losses = []

    for _ in range(iterations):
        optimizer.zero_grad()
        loss = loss_func(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        iterates.append(x.data.numpy().copy())

    loss = loss_func(x)
    losses.append(loss.item())
    return losses, iterates


if __name__ == "__main__":

    x_0, x_star, loss_func, constraint = setup_problem(make_nonconvex=False)
    iterations = 10
    losses_all = {}
    iterates_all = {}
    for opt_class in OPTIMIZER_CLASSES:
        losses_, iterates_ = optimize(x_0,
                                      loss_func,
                                      constraint,
                                      opt_class,
                                      iterations)
        losses_all[opt_class.name] = losses_
        iterates_all[opt_class.name] = iterates_
    # print(losses)
    fig, ax = plt.subplots()
    for opt_class in OPTIMIZER_CLASSES:
        ax.plot(np.arange(iterations + 1), losses_all[opt_class.name],
                label=opt_class.name)
    fig.legend()
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for ax, opt_class in zip(axes.reshape(-1), OPTIMIZER_CLASSES):
        ax.plot(*zip(*iterates_all[opt_class.name]), '-o', label=opt_class.name, alpha=.6)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.legend(loc='lower left')
    plt.show()
