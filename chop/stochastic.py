"""This module contains stochastic first order optimizers.
These are meant to be used in replacement of optimizers such as SGD, Adam etc,
for training a model over batches of a dataset."""

import warnings

import torch
from torch.optim import Optimizer

import numpy as np


EPS = np.finfo(np.float32).eps


def backtracking_step_size(
    x,
    f_t,
    old_f_t,
    f_grad,
    certificate,
    lipschitz_t,
    max_step_size,
    update_direction,
    norm_update_direction,
):
    """Backtracking step-size finding routine for FW-like algorithms

    Args:
        x: array-like, shape (n_features,)
            Current iterate
        f_t: float
            Value of objective function at the current iterate.
        old_f_t: float
            Value of objective function at previous iterate.
        f_grad: callable
            Callable returning objective function and gradient at
            argument.
        certificate: float
            FW gap
        lipschitz_t: float
            Current value of the Lipschitz estimate.
        max_step_size: float
            Maximum admissible step-size.
        update_direction: array-like, shape (n_features,)
            Update direction given by the FW variant.
        norm_update_direction: float
            Squared L2 norm of update_direction
    Returns:
        step_size_t: float
            Step-size to be used to compute the next iterate.
        lipschitz_t: float
            Updated value for the Lipschitz estimate.
        f_next: float
            Objective function evaluated at x + step_size_t d_t.
        grad_next: array-like
            Gradient evaluated at x + step_size_t d_t.
    """
    ratio_decrease = 0.9
    ratio_increase = 2.0
    max_ls_iter = 100
    if old_f_t is not None:
        tmp = (certificate ** 2) / (2 * (old_f_t - f_t) * norm_update_direction)
        lipschitz_t = max(min(tmp, lipschitz_t), lipschitz_t * ratio_decrease)
    for _ in range(max_ls_iter):
        step_size_t = certificate / (norm_update_direction * lipschitz_t)
        if step_size_t < max_step_size:
            rhs = -0.5 * step_size_t * certificate
        else:
            step_size_t = max_step_size
            rhs = (
                -step_size_t * certificate
                + 0.5 * (step_size_t ** 2) * lipschitz_t * norm_update_direction
            )
        f_next, grad_next = f_grad(x + step_size_t * update_direction)
        if f_next - f_t <= rhs + EPS:
            # .. sufficient decrease condition verified ..
            break
        else:
            lipschitz_t *= ratio_increase
    else:
        warnings.warn(
            "Exhausted line search iterations in minimize_frank_wolfe", RuntimeWarning
        )
    return step_size_t, lipschitz_t, f_next, grad_next


class PGD(Optimizer):
    """Projected Gradient Descent"""
    name = 'PGD'

    def __init__(self, params, constraint, lr=.1):
        self.prox = lambda x: constraint.prox(x.unsqueeze(0)).squeeze()
        if not (type(lr) == float or lr == 'sublinear'):
            raise ValueError("lr must be float or 'sublinear'.")
        self.lr = lr

        defaults = dict(prox=self.prox, name=self.name)
        super(PGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for groups in self.param_groups:
            for p in groups['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'We do not yet support sparse gradients.')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0.
                state['step'] += 1.

                if self.lr == 'sublinear':
                    step_size = 1. / (state['step'] + 1.)
                else:
                    step_size = self.lr

                p.copy_(self.prox(p - step_size * grad))
        return loss


class PGDMadry(Optimizer):
    """What Madry et al. call PGD"""
    name = 'PGD-Madry'

    def __init__(self, params, constraint, lr):
        self.prox = lambda x: constraint.prox(x.unsqueeze(0)).squeeze()

        def _lmo(u, x):
            update_direction, max_step_size = constraint.lmo(u.unsqueeze(0), x.unsqueeze(0))
            return update_direction.squeeze(dim=0), max_step_size
        self.lmo = _lmo

        if not (type(lr) == float or lr == 'sublinear'):
            raise ValueError("lr must be float or 'sublinear'.")
        self.lr = lr
        defaults = dict(prox=self.prox, lmo=self.lmo, name=self.name)
        super(PGDMadry, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, step_size=None, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for groups in self.param_groups:
            for p in groups['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'We do not yet support sparse gradients.')
                # Keep track of the step
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0.
                state['step'] += 1.

                if self.lr == 'sublinear':
                    step_size = 1. / (state['step'] + 1.)
                else:
                    step_size = self.lr
                lmo_res, _ = self.lmo(-p.grad, p)
                normalized_grad = lmo_res + p
                p.copy_(self.prox(p + step_size * normalized_grad))
        return loss


class PairwiseFrankWolfe(Optimizer):
    """Pairwise Frank-Wolfe algorithm"""
    name = "Pairwise-FW"

    def __init__(self, params, constraint, lr=.1, momentum=.9):
        if not (type(lr) == float or lr == 'sublinear'):
            raise ValueError("lr must be float or 'sublinear'.")

        def _lmo(u, x):
            update_direction, max_step_size = constraint.lmo_pairwise(u.unsqueeze(0), x.unsqueeze(0))
            return update_direction.squeeze(dim=0), max_step_size
        self.lmo = _lmo
        self.lr = lr
        self.momentum = momentum
        defaults = dict(lmo=self.lmo, name=self.name, lr=self.lr, momentum=self.momentum)
        super(PairwiseFrankWolfe, self).__init__(params, defaults)


class MomentumFrankWolfe(Optimizer):
    """Class for the Stochastic Frank-Wolfe algorithm given in Mokhtari et al.
    This is essentially FrankWolfe with Momentum.
    We use the tricks from [Pokutta, Spiegel, Zimmer, 2020].
    https://arxiv.org/abs/2010.07243"""
    name = 'Momentum-FW'

    def __init__(self, params, constraint, lr=.1, momentum=.9):
        def _lmo(u, x):
            update_direction, max_step_size = constraint.lmo(u.unsqueeze(0), x.unsqueeze(0))
            return update_direction.squeeze(dim=0), max_step_size
        self.lmo = _lmo

        if type(lr) == float:
            if not (0. <= lr <= 1.):
                raise ValueError("lr must be in [0., 1.].")
        self.lr = lr
        if not(0. <= momentum <= 1.):
            raise ValueError("Momentum must be in [0., 1.].")
        self.momentum = momentum
        defaults = dict(lmo=self.lmo, name=self.name, lr=self.lr, momentum=self.momentum)
        super(MomentumFrankWolfe, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss
            """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'SFW does not yet support sparse gradients.')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_estimate'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                if self.lr == 'sublinear':
                    step_size = 1. / (state['step'] + 1.)
                elif type(self.lr) == float:
                    step_size = self.lr
                else:
                    raise ValueError("lr must be float or 'sublinear'.")

                if self.momentum is None:
                    rho = (1. / (state['step'] + 1)) ** (1/3)
                    momentum = 1. - rho
                else:
                    momentum = self.momentum

                state['step'] += 1.

                state['grad_estimate'] += (1. - momentum) * (grad - state['grad_estimate'])
                grad_norm = torch.norm(state['grad_estimate'])
                update_direction, _ = self.lmo(-state['grad_estimate'], p)
                step_size = min(1., step_size * grad_norm / torch.norm(update_direction))
                p += step_size * update_direction
        return loss
