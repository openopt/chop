import torch
from torch.optim import Optimizer


"""This API is inspired by the COPT project
https://github.com/openopt/copt"""
class PGD(Optimizer):
    """Projected Gradient Descent"""
    name = 'PGD'

    def __init__(self, params, constraint):
        self.prox = constraint.prox
        defaults = dict(prox=self.prox, name=self.name)
        super(PGD, self).__init__(params, defaults)

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
                if not step_size:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0.
                    state['step'] += 1.
                    step_size = 1. / (state['step'] + 1.)
                p.add_(self.prox(p - step_size * grad) - p)
        return loss


class PGDMadry(Optimizer):
    """What Madry et al. call PGD"""
    name = 'PGD-Madry'

    def __init__(self, params, constraint):
        self.prox = constraint.prox
        self.lmo = constraint.lmo
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
                if not step_size:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0.
                    state['step'] += 1.
                    step_size = 1. / (state['step'] + 1.)
                lmo_res, _ = self.lmo(-p.grad, p)
                normalized_grad = lmo_res + p
                p.add_(self.prox(p + step_size * normalized_grad) - p)
        return loss


# TODO: Try rescaling the update_direction as alpha * \|gradient\|
# + Demyanov Rubinov step-size
class FrankWolfe(Optimizer):
    """Vanilla Frank-Wolfe algorithm"""
    name = 'Vanilla-FW'

    def __init__(self, params, constraint):
        self.lmo = constraint.lmo
        defaults = dict(lmo=self.lmo, name=self.name)
        super(FrankWolfe, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, step_size=None, closure=None):
        """Performs a single optimization step

        Arguments:
            step_size: Ignored by this optimizer
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss"""

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
                        'FW does not yet support sparse gradients.')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0.

                state['step'] += 1.

                if step_size is None:
                    step_size = 1. / (state['step'] + 1)

                update_direction, _ = self.lmo(-p.grad, p)
                p += (2. / (state['step'] + 2.)) * update_direction
        return loss


class MomentumFrankWolfe(Optimizer):
    """Class for the Stochastic Frank-Wolfe algorithm given in Mokhtari et al.
    This is essentially FrankWolfe with Momentum."""
    name = 'Momentum-FW'

    def __init__(self, params, constraint):
        self.lmo = constraint.lmo
        defaults = dict(lmo=self.lmo, name=self.name)
        super(MomentumFrankWolfe, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, step_size=None, momentum=None, closure=None):
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

                state['step'] += 1.

                if step_size is None:
                    step_size = 1. / (state['step'] + 1)
                if momentum is None:
                    momentum = (1. / (state['step'] + 1)) ** (1/3)

                state['grad_estimate'] += (1. - momentum) * (grad - state['grad_estimate'])
                update_direction, _ = self.lmo(-state['grad_estimate'], p)
                p += step_size * update_direction
        return loss
