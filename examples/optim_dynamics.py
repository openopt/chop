"""Sets up simple 2-d problems on Linf balls to see dynamics of different constrained optimization algorithms."""
from functools import partial
import matplotlib.pyplot as plt
import torch

from constopt.constraints import LinfBall
from constopt.optim import minimize_pgd, minimize_pgd_madry
from constopt import utils

torch.random.manual_seed(0)

OPTIMIZERS = [minimize_pgd, minimize_pgd_madry]


def setup_problem(make_nonconvex=False):
    alpha = 1.
    x_star = torch.tensor([alpha, alpha/2]).unsqueeze(0)
    x_0 = torch.zeros_like(x_star)

    @utils.closure
    def loss_func(x):
        val = .5 * ((x - x_star) ** 2).sum()
        if make_nonconvex:
            val += .1 * torch.sin(50 * torch.norm(x, p=1) + .1)
        return val

    constraint = LinfBall(alpha)

    return x_0, x_star, loss_func, constraint


def log(kwargs, iterates, losses):
    x= kwargs['x'].squeeze().data
    iterates.append(x)
    val = kwargs['closure'](x, return_jac=False).data
    losses.append(val)


if __name__ == "__main__":

    x_0, x_star, loss_func, constraint = setup_problem(make_nonconvex=False)
    iterations = 10

    iterates_pgd = [x_0.squeeze().data]
    iterates_pgd_madry = [x_0.squeeze().data]
    losses_pgd = [loss_func(x_0, return_jac=False).data]
    losses_pgd_madry = [loss_func(x_0, return_jac=False).data]

    log_pgd = partial(log, iterates=iterates_pgd, losses=losses_pgd)
    log_pgd_madry = partial(log, iterates=iterates_pgd_madry, losses=losses_pgd_madry)

    sol_pgd = minimize_pgd(loss_func, x_0, constraint.prox,
                                 step=None,
                                 max_iter=iterations,
                                 callback=log_pgd)

    sol_pgd_madry = minimize_pgd_madry(loss_func, x_0, constraint.prox,
                                       constraint.lmo,
                                       step=None,
                                       max_iter=iterations,
                                       callback=log_pgd_madry)

    fig, ax = plt.subplots()
    ax.plot(losses_pgd, label="PGD")
    ax.plot(losses_pgd_madry, label="PGD Madry")
    fig.legend()
    fig.savefig("examples/plots/optim/2-D_Linf_losses.png")

    fig, ax = plt.subplots(figsize=(8, 4), ncols=2, sharex=True, sharey=True)
    ax[0].plot(*zip(*iterates_pgd), '-o', label="PGD", alpha=.6)
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)
    ax[0].legend()

    ax[1].plot(*zip(*iterates_pgd_madry), '-o', label="PGD Madry", alpha=.6)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].legend()

    fig.savefig("examples/plots/optim/2-D_Linf_iterates.png")