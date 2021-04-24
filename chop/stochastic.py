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
        tmp = (certificate ** 2) / \
            (2 * (old_f_t - f_t) * norm_update_direction)
        lipschitz_t = max(min(tmp, lipschitz_t), lipschitz_t * ratio_decrease)
    for _ in range(max_ls_iter):
        step_size_t = certificate / (norm_update_direction * lipschitz_t)
        if step_size_t < max_step_size:
            rhs = -0.5 * step_size_t * certificate
        else:
            step_size_t = max_step_size
            rhs = (
                -step_size_t * certificate
                + 0.5 * (step_size_t ** 2) *
                lipschitz_t * norm_update_direction
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
    """Proximal Gradient Descent

    Args:
      params: [torch.Parameter]
        List of parameters to optimize over
      prox: [callable or None]
        List of prox operators, one per parameter.
      lr: float
        Learning rate
      momentum: float in [0, 1]

      normalization: str
        Type of gradient normalization to be used.
        Possible values are 'none', 'L2', 'Linf', 'sign'.

    """
    name = 'PGD'
    POSSIBLE_NORMALIZATIONS = {'none', 'L2', 'Linf', 'sign'}

    def __init__(self, params, prox=None, lr=.1, momentum=.9, normalization='none'):
        params = list(params)
        if prox is None:
            prox = [None] * len(params)

        self.prox = []
        for prox_el in prox:
            if prox_el is not None:
                self.prox.append(lambda x, s=None: prox_el(
                    x.unsqueeze(0), s).squeeze(0))
            else:
                self.prox.append(lambda x, s=None: x)

        if not (type(lr) == float or lr == 'sublinear'):
            raise ValueError(f"lr must be float or 'sublinear', got {lr}.")
        self.lr = lr

        if not(0. <= momentum <= 1.):
            raise ValueError("Momentum must be in [0., 1.].")
        self.momentum = momentum

        if normalization in self.POSSIBLE_NORMALIZATIONS:
            self.normalization = normalization
        else:
            raise ValueError(
                f"Normalization must be in {self.POSSIBLE_NORMALIZATIONS}")
        defaults = dict(prox=self.prox, name=self.name,
                        momentum=self.momentum, lr=self.lr,
                        normalization=self.normalization)
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
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        'We do not yet support sparse gradients.')

                state = self.state[p]
                # Initialization
                if len(state) == 0:
                    state['step'] = 0.
                    state['grad_estimate'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                state['step'] += 1.
                state['grad_estimate'].add_(
                    grad - state['grad_estimate'], alpha=1. - self.momentum)

                grad_est = normalize_gradient(
                    state['grad_estimate'], group['normalization'])

                if group['lr'] == 'sublinear':
                    state['lr'] = 1. / (state['step'] + 1.)
                else:
                    state['lr'] = group['lr']

                new_p = self.prox[idx](p - state['lr'] * grad_est, 1.)
                state['certificate'] = torch.norm((p - new_p) / state['lr'])
                p.copy_(new_p)
                idx += 1
        return loss


class PGDMadry(Optimizer):
    """PGD from [1]. 

    Args:
      params: [torch.Tensor]
        list of parameters to optimize

      lmo: [callable]
        list of lmo operators for each parameter

      prox: [callable or None] or None
        list of prox operators for each parameter

      lr: float > 0
        learning rate

    References:
      Madry, Aleksander, and Makelov, Aleksandar, and Schmidt, Ludwig,
      and Tsipras, Dimitris, and Vladu, Adrian. Towards Deep Learning Models
      Resistant to Adversarial Attacks. ICLR 2018.
    """
    name = 'PGD-Madry'

    def __init__(self, params, lmo, prox=None, lr=1e-2):
        self.prox = []
        for prox_el in prox:
            if prox_el is None:
                def prox_el(x, s=None):
                    return x

            def _prox(x, s=None):
                return prox_el(x.unsqueeze(0), s).squeeze()
            self.prox.append(_prox)

        self.lmo = []
        for lmo_el in lmo:
            def _lmo(u, x):
                update_direction, max_step_size = lmo_el(
                    u.unsqueeze(0), x.unsqueeze(0))
                return update_direction.squeeze(dim=0), max_step_size
            self.lmo.append(_lmo)

        if not (type(lr) == float or lr == 'sublinear'):
            raise ValueError("lr must be float or 'sublinear'.")

        defaults = dict(prox=self.prox, lmo=self.lmo, lr=lr,
                        name=self.name)
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
            idx = 0
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

                if state['lr'] == 'sublinear':
                    step_size = 1. / (state['step'] + 1.)
                else:
                    step_size = state['lr']
                lmo_res, _ = self.lmo[idx](-p.grad, p)
                normalized_grad = lmo_res + p
                new_p = self.prox[idx](p + step_size * normalized_grad)
                state['certificate'] = torch.norm((p - new_p) / step_size)
                p.copy_(new_p)
                idx += 1
        return loss


class S3CM(Optimizer):
    """
    Stochastic Three Composite Minimization (S3CM)

    Args:
      params: [torch.Tensor]
        list of parameters to optimize

      prox1: [callable or None] or None
        Proximal operator for first constraint set.

      prox2: [callable or None] or None
        Proximal operator for second constraint set.

      lr: float > 0
        Learning rate

      normalization: str in {'none', 'L2', 'Linf', 'sign'}
        Normalizes the gradient. 'L2', 'Linf' divide the gradient by the corresponding norm.
        'sign' uses the sign of the gradient.

    References:
      Yurtsever, Alp, and Vu, Bang Cong, and Cevher, Volkan.
      "Stochastic Three-Composite Convex Minimization" NeurIPS 2016
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
            raise ValueError(
                f"Normalization must be in {self.POSSIBLE_NORMALIZATIONS}")

        if prox1 is None:
            prox1 = [None] * len(params)
        if prox2 is None:
            prox2 = [None] * len(params)

        self.prox1 = []
        self.prox2 = []

        for prox1_, prox2_ in zip(prox1, prox2):
            if prox1_ is None:
                def prox1_(x, s=None): return x

            if prox2_ is None:
                def prox2_(x, s=None): return x

            self.prox1.append(lambda x, s=None: prox1_(
                x.unsqueeze(0), s).squeeze(dim=0))
            self.prox2.append(lambda x, s=None: prox2_(
                x.unsqueeze(0), s).squeeze(dim=0))

        defaults = dict(lr=self.lr, prox1=self.prox1, prox2=self.prox2,
                        normalization=self.normalization)
        super(S3CM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
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
                    state['iterate_2'] = self.prox2[idx](p, self.lr)
                    state['dual'] = (state['iterate_1'] -
                                     state['iterate_2']) / self.lr

                state['iterate_2'] = self.prox2[idx](
                    state['iterate_1'] + self.lr * state['dual'], self.lr)
                state['dual'].add_(
                    (state['iterate_1'] - state['iterate_2']) / self.lr)
                state['iterate_1'] = self.prox1[idx](state['iterate_2']
                                                     - self.lr * (grad + state['dual']), self.lr)

                p.copy_(state['iterate_2'])
                idx += 1
        return loss


class PairwiseFrankWolfe(Optimizer):
    """Pairwise Frank-Wolfe algorithm"""
    name = "Pairwise-FW"

    def __init__(self, params, lmo_pairwise, lr=.1, momentum=.9):
        if not (type(lr) == float or lr == 'sublinear'):
            raise ValueError("lr must be float or 'sublinear'.")

        def _lmo(u, x):
            update_direction, max_step_size = lmo_pairwise(
                u.unsqueeze(0), x.unsqueeze(0))
            return update_direction.squeeze(dim=0), max_step_size
        self.lmo = _lmo
        self.lr = lr
        self.momentum = momentum
        defaults = dict(lmo=self.lmo, name=self.name,
                        lr=self.lr, momentum=self.momentum)
        super(PairwiseFrankWolfe, self).__init__(params, defaults)

        raise NotImplementedError


class FrankWolfe(Optimizer):
    """Class for the Stochastic Frank-Wolfe algorithm given in Mokhtari et al.
    This is essentially Frank-Wolfe with Momentum.
    We use the tricks from [1] for gradient normalization.

    Args:
      params: [torch.Tensor]
        Parameters to optimize over.

      lmo: [callable]
        List of LMO operators.

      lr: float
        Learning rate

      momentum: float in [0, 1]
        Amount of momentum to be used in gradient estimator

      weight_decay: float > 0
        Amount of L2 regularization to be added

      normalization: str in {'gradient', 'none'}
        Gradient normalization to be used. 'gradient' option is described in [1].

    References:
      Pokutta, Sebastian, and Spiegel, Christoph and Zimmer, Max,
      Deep Neural Network Training with Frank Wolfe. 2020.
    """
    name = 'Frank-Wolfe'
    POSSIBLE_NORMALIZATIONS = {'gradient', 'none'}

    def __init__(self, params, lmo, lr=.1, momentum=0.,
                 weight_decay=0.,
                 normalization='none'):

        lmo_candidates = []
        for oracle in lmo:
            if oracle is None:
                # Then FW will not be used on this parameter
                _lmo = None
            else:
                def _lmo(u, x):
                    update_direction, max_step_size = oracle(
                        u.unsqueeze(0), x.unsqueeze(0))
                    return update_direction.squeeze(dim=0), max_step_size
            lmo_candidates.append(_lmo)

        self.lmo = []
        useable_params = []
        for param, oracle in zip(params, lmo):
            if oracle:
                useable_params.append(param)
                self.lmo.append(oracle)
            else:
                msg = (f"No LMO was provided for parameter {param}. "
                       f"Frank-Wolfe will not optimize this parameter. "
                       f"Please use another optimizer.")
                warnings.warn(msg)

        if type(lr) == float:
            if not (0. < lr <= 1.):
                raise ValueError("lr must be in (0., 1.].")
        self.lr = lr
        if type(momentum) == float:
            if not(0. <= momentum <= 1.):
                raise ValueError("Momentum must be in [0., 1.].")
        self.momentum = momentum
        if not (weight_decay >= 0):
            raise ValueError("weight_decay should be nonnegative.")
        self.weight_decay = weight_decay
        if normalization not in self.POSSIBLE_NORMALIZATIONS:
            raise ValueError(
                f"Normalization must be in {self.POSSIBLE_NORMALIZATIONS}.")
        self.normalization = normalization
        defaults = dict(lmo=self.lmo, name=self.name, lr=self.lr,
                        momentum=self.momentum,
                        weight_decay=weight_decay,
                        normalization=self.normalization)
        super(FrankWolfe, self).__init__(useable_params, defaults)

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
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad + self.weight_decay * p
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

                if self.momentum is None or self.momentum == 'sublinear':
                    rho = (1. / (state['step'] + 1)) ** (1/3)
                    momentum = 1. - rho
                else:
                    momentum = self.momentum

                state['step'] += 1.

                state['grad_estimate'].add_(
                    grad - state['grad_estimate'], alpha=1. - momentum)
                update_direction, _ = self.lmo[idx](-state['grad_estimate'], p)
                state['certificate'] = (-state['grad_estimate']
                                        * update_direction).sum()
                if group['normalization'] == 'gradient':
                    grad_norm = torch.norm(state['grad_estimate'])
                    step_size = min(1., step_size * grad_norm /
                                    torch.linalg.norm(update_direction))
                elif group['normalization'] == 'none':
                    pass
                p.add_(step_size * update_direction)
                idx += 1
        return loss


class SplittingProxFW(Optimizer):
    # TODO: write docstring!

    name = 'Hybrid Prox FW Splitting'

    POSSIBLE_NORMALIZATIONS = {'none', 'gradient'}

    def __init__(self, params, lmo, prox=None,
                 lr_lmo=.1,
                 lr_prox=.1,
                 momentum=0., weight_decay=0.,
                 normalization='none'):
        params = list(params)
        # initialize proxes
        if prox is None:
            prox = [None] * len(params)

        prox_candidates = []

        def prox_maker(oracle):
            if oracle:
                def _prox(x, s=None):
                    return oracle(x.unsqueeze(0), s).squeeze(0)
            else:
                def _prox(x, s=None):
                    return x, s
            return _prox

        prox_candidates = [prox_maker(oracle) for oracle in prox]
        # initialize lmos

        def lmo_maker(oracle):
            def _lmo(u, x):
                update_direction, max_step_size = oracle(
                    u.unsqueeze(0), x.unsqueeze(0))
                return update_direction.squeeze(dim=0), max_step_size.squeeze(dim=0)

            return _lmo

        lmo_candidates = [lmo_maker(oracle) if oracle else None for oracle in lmo]

        self.lmo = []
        self.prox = []
        useable_params = []
        for param, lmo_oracle, prox_oracle in zip(params, lmo_candidates, prox_candidates):
            if lmo_oracle is not None:
                useable_params.append(param)
                self.lmo.append(lmo_oracle)
                self.prox.append(prox_oracle)
            else:
                msg = (f"No LMO was provided for parameter {param}. "
                       f"This optimizer will not optimize this parameter. "
                       f"Please pass this parameter to another optimizer.")
                warnings.warn(msg)

        for name, lr in (('lr_lmo', lr_lmo),
                         ('lr_prox', lr_prox)):
            if not ((type(lr) == float) or lr == 'sublinear'):
                msg = f"{name} should be a float or 'sublinear', got {lr}."
                raise ValueError(msg)

        if (momentum != 'sublinear') and (not (0. <= momentum <= 1.)):
            raise ValueError("momentum must be in [0., 1.] or 'sublinear'.")

        if not (weight_decay >= 0):
            raise ValueError("weight_decay must be nonnegative.")
        self.weight_decay = weight_decay

        if normalization not in self.POSSIBLE_NORMALIZATIONS:
            raise ValueError(
                f"Normalization must be in {self.POSSIBLE_NORMALIZATIONS}")
        defaults = dict(lmo=self.lmo, prox=self.prox,
                        name=self.name,
                        momentum=momentum,
                        lr_lmo=lr_lmo,
                        lr_prox=lr_prox,
                        weight_decay=weight_decay,
                        normalization=normalization)

        super(SplittingProxFW, self).__init__(useable_params, defaults)

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
            idx = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if grad.is_sparse:
                    msg = "We do not yet support sparse gradients."
                    raise RuntimeError(msg)
                # Keep track of the step
                grad += group['weight_decay'] * p

                # Initialization
                if len(state) == 0:
                    state['step'] = 0.
                    # split variable: p = x + y
                    state['x'] = .5 * p.detach().clone()
                    state['y'] = .5 * p.detach().clone()
                    # initialize grad estimate
                    state['grad_est'] = torch.zeros_like(p)
                    # initialize learning rates
                    state['lr_prox'] = group['lr_prox'] if type(
                        group['lr_prox'] == float) else 0.
                    state['lr_lmo'] = group['lr_lmo'] if type(
                        group['lr_lmo'] == float) else 0.
                    state['momentum'] = group['momentum'] if type(
                        group['momentum'] == float) else 0.

                for lr in ('lr_prox', 'lr_lmo'):
                    if group[lr] == 'sublinear':
                        state[lr] = 2. / (state['step'] + 2)

                if group['momentum'] == 'sublinear':
                    rho = 4. / (state['step'] + 8.) ** (2/3)
                    state['momentum'] = 1. - rho

                state['step'] += 1.
                state['grad_est'].add_(
                    grad - state['grad_est'], alpha=1. - state['momentum'])

                y_update, max_step_size = group['lmo'][idx](
                    -state['grad_est'], state['y'])

                state['lr_lmo'] = min(max_step_size, state['lr_lmo'])

                if group['normalization'] == 'gradient':
                    # Normalize LMO update direction
                    grad_norm = torch.linalg.norm(state['grad_est'])
                    y_update *= min(1, grad_norm / torch.linalg.norm(y_update))
                state['lr_lmo'] = min(state['lr_lmo'], max_step_size)
                w = y_update + state['y']
                v = group['prox'][idx](
                    state['x'] + state['y'] - w - state['grad_est'] / state['lr_prox'], state['lr_prox'])
                x_update = v - state['x']

                state['y'].add_(y_update, alpha=state['lr_lmo'])
                state['x'].add_(x_update, alpha=state['lr_lmo'])

                p.copy_(state['x'] + state['y'])
                idx += 1
        return loss
