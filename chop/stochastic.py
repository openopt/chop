"""
Stochastic optimizers.
=========================

This module contains stochastic first order optimizers.
These are meant to be used in replacement of optimizers such as SGD, Adam etc,
for training a model over batches of a dataset.
The API in this module is inspired by torch.optim.

"""

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


def normalize_gradient(grad, normalization):
    if normalization == 'none':
        return grad
    elif normalization == 'Linf':
        grad = grad / abs(grad).max()

    elif normalization == 'sign':
        grad = torch.sign(grad)

    elif normalization == 'L2':
        grad = grad / torch.norm(grad)

    return grad
        


class PGD(Optimizer):
    """Projected Gradient Descent"""
    name = 'PGD'
    POSSIBLE_NORMALIZATIONS = {'none', 'L2', 'Linf', 'sign'}
    
    def __init__(self, params, constraint, lr=.1, normalization='none'):
        self.prox = lambda x: constraint.prox(x.unsqueeze(0)).squeeze()
        if not (type(lr) == float or lr == 'sublinear'):
            raise ValueError("lr must be float or 'sublinear'.")
        self.lr = lr
        if normalization in self.POSSIBLE_NORMALIZATIONS:
            self.normalization = normalization
        else:
            raise ValueError(f"Normalization must be in {self.POSSIBLE_NORMALIZATIONS}")
        defaults = dict(prox=self.prox, name=self.name, normalization=self.normalization)
        super(PGD, self).__init__(params, defaults)

    @property
    @torch.no_grad()
    def certificate(self):
        """A generator over the current convergence certificate estimate
        for each optimized parameter."""
        for groups in self.param_groups:
            for p in groups['params']:
                state = self.state[p]
                yield state['certificate']

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

                grad = normalize_gradient(grad, self.normalization)
                
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

                new_p = self.prox(p - step_size * grad)
                state['certificate'] = torch.norm((p - new_p) / step_size)
                p.copy_(new_p)
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

    @property
    @torch.no_grad()
    def certificate(self):
        """A generator over the current convergence certificate estimate
        for each optimized parameter."""
        for groups in self.param_groups:
            for p in groups['params']:
                state = self.state[p]
                yield state['certificate']

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
                new_p = self.prox(p + step_size * normalized_grad)
                state['certificate'] = torch.norm((p - new_p) / step_size)
                p.copy_(new_p)
        return loss


class S3CM(Optimizer):
    """Stochastic Three Composite Minimization (S3CM)
    Cf
    https://arxiv.org/abs/1701.09033
    Yurtsever, Vu, Cevher, 2017
    """
    name = "S3CM"
    POSSIBLE_NORMALIZATIONS = {'none', 'L2', 'Linf', 'sign'}

    def __init__(self, params, prox1=None, prox2=None, lr=.1, normalization='none'):
        if not type(lr) == float:
            raise ValueError("lr must be a float.")

        self.lr = lr
        if normalization in self.POSSIBLE_NORMALIZATIONS:
            self.normalization = normalization
        else:
            raise ValueError(f"Normalization must be in {self.POSSIBLE_NORMALIZATIONS}")

        if prox1 is None:
            def prox1(x, s=None): return x

        if prox2 is None:
            def prox2(x, s=None): return x

        self.prox1 = lambda x, s: prox1(x.unsqueeze(0), s).squeeze(dim=0)
        self.prox2 = lambda x, s: prox2(x.unsqueeze(0), s).squeeze(dim=0)

        defaults = dict(lr=self.lr, prox1=self.prox1, prox2=self.prox2,
                        normalization=self.normalization)
        super(S3CM, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                grad = normalize_gradient(grad, self.normalization)

                if grad.is_sparse:
                    raise RuntimeError(
                        'S3CM does not yet support sparse gradients.')
                state = self.state[p]
                # initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['iterate_1'] = p.clone().detach()
                    state['iterate_2'] = self.prox2(p, self.lr)
                    state['dual'] = (state['iterate_1'] - state['iterate_2']) / self.lr

                state['iterate_2'] = self.prox2(state['iterate_1'] + self.lr * state['dual'], self.lr)
                state['dual'].add_((state['iterate_1'] - state['iterate_2']) / self.lr)
                state['iterate_1'] = self.prox1(state['iterate_2'] 
                                                - self.lr * (grad + state['dual']), self.lr)

                p.copy_(state['iterate_2'])
           
        

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

        raise NotImplementedError


class FrankWolfe(Optimizer):
    """Class for the Stochastic Frank-Wolfe algorithm given in Mokhtari et al.
    This is essentially FrankWolfe with Momentum.
    We use the tricks from [Pokutta, Spiegel, Zimmer, 2020].
    https://arxiv.org/abs/2010.07243"""
    name = 'Frank-Wolfe'
    POSSIBLE_NORMALIZATIONS = {'gradient', 'none'}

    def __init__(self, params, constraint, lr=.1, momentum=.9, normalization='none'):
        def _lmo(u, x):
            update_direction, max_step_size = constraint.lmo(u.unsqueeze(0), x.unsqueeze(0))
            return update_direction.squeeze(dim=0), max_step_size
        self.lmo = _lmo

        if type(lr) == float:
            if not (0. < lr <= 1.):
                raise ValueError("lr must be in (0., 1.].")
        self.lr = lr
        if type(momentum) == float:
            if not(0. <= momentum <= 1.):
                raise ValueError("Momentum must be in [0., 1.].")
        self.momentum = momentum
        if normalization not in self.POSSIBLE_NORMALIZATIONS:
            raise ValueError(f"Normalization must be in {self.POSSIBLE_NORMALIZATIONS}.")
        self.normalization = normalization
        defaults = dict(lmo=self.lmo, name=self.name, lr=self.lr, 
                        momentum=self.momentum, normalization=self.normalization)
        super(FrankWolfe, self).__init__(params, defaults)

    @property
    @torch.no_grad()
    def certificate(self):
        """A generator over the current convergence certificate estimate
        for each optimized parameter."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                yield state['certificate']

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

                state['grad_estimate'].add_(grad - state['grad_estimate'], alpha=1. - momentum)
                grad_norm = torch.norm(state['grad_estimate'])
                update_direction, _ = self.lmo(-state['grad_estimate'], p)
                state['certificate'] = torch.dot(-state['grad_estimate'], update_direction)
                if self.normalization == 'gradient':
                    step_size = min(1., step_size * grad_norm / torch.norm(update_direction))
                elif self.normalization == 'none':
                    pass
                p += step_size * update_direction
        return loss
