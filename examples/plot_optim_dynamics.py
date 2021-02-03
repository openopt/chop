"""
Full-gradient constrained optimization dynamics.
================================================
Sets up simple 2-d problems on Linf balls to visualize dynamics of various
constrained optimization algorithms.
"""
from functools import partial
import matplotlib.pyplot as plt
import torch

from chop.constraints import LinfBall
from chop.optim import minimize_frank_wolfe, minimize_pgd, minimize_pgd_madry, minimize_three_split
from chop import utils 

torch.random.manual_seed(0)


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

    x_0, x_star, loss_func, constraint = setup_problem(make_nonconvex=True)
    iterations = 10

    iterates_pgd = [x_0.squeeze().data]
    iterates_pgd_madry = [x_0.squeeze().data]
    iterates_splitting = [x_0.squeeze().data]
    iterates_fw = [x_0.squeeze().data]

    losses_pgd = [loss_func(x_0, return_jac=False).data]
    losses_pgd_madry = [loss_func(x_0, return_jac=False).data]
    losses_splitting = [loss_func(x_0, return_jac=False).data]
    losses_fw = [loss_func(x_0, return_jac=False).data]

    log_pgd = partial(log, iterates=iterates_pgd, losses=losses_pgd)
    log_pgd_madry = partial(log, iterates=iterates_pgd_madry, losses=losses_pgd_madry)
    log_splitting = partial(log, iterates=iterates_splitting, losses=losses_splitting)
    log_fw = partial(log, iterates=iterates_fw, losses=losses_fw)

    sol_pgd = minimize_pgd(loss_func, x_0, constraint.prox,
                           max_iter=iterations,
                           callback=log_pgd)

    sol_pgd_madry = minimize_pgd_madry(loss_func, x_0, constraint.prox,
                                       constraint.lmo,
                                       step=2. / iterations,
                                       max_iter=iterations,
                                       callback=log_pgd_madry)

    sol_splitting = minimize_three_split(loss_func, x_0, prox1=constraint.prox, 
                                         max_iter=iterations, callback=log_splitting)

    sol_fw = minimize_frank_wolfe(loss_func, x_0, constraint.lmo, callback=log_fw,
                                  max_iter=iterations)

    fig, ax = plt.subplots()
    ax.plot(losses_pgd, label="PGD")
    ax.plot(losses_pgd_madry, label="PGD Madry")
    ax.plot(losses_splitting, label="Operator Splitting")
    ax.plot(losses_fw, label="Frank-Wolfe")
    fig.legend()
    plt.show()

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].plot(*zip(*iterates_pgd), '-o', label="PGD", alpha=.6)
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)
    ax[0].legend()

    ax[1].plot(*zip(*iterates_pgd_madry), '-o', label="PGD Madry", alpha=.6)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].legend()

    ax[2].plot(*zip(*iterates_splitting), '-o', label="Operator Splitting", alpha=.6)
    ax[2].set_xlim(-1, 1)
    ax[2].set_ylim(-1, 1)
    ax[2].legend()

    ax[3].plot(*zip(*iterates_fw), '-o', label="Frank-Wolfe", alpha=.6)
    ax[3].set_xlim(-1, 1)
    ax[3].set_ylim(-1, 1)
    ax[3].legend()

    plt.show()
