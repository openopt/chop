import torch
from torch.optim import Optimizer


"""This API is inspired by the COPT project
https://github.com/openopt/copt"""


class FrankWolfe(Optimizer):
    """Vanilla Frank-Wolfe algorithm"""

    def __init__(self, params, lmo):
        self.lmo = lmo
        defaults = dict(lmo=lmo)
        super(FrankWolfe, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step
        
        Arguments:
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

                update_direction, _ = self.lmo(p.grad, p)
                p += (2. / (state['step'] + 2.)) * update_direction
        return loss


class StochasticFrankWolfe(Optimizer):
    """Class for the Stochastic Frank-Wolfe algorithm given in Mokhtari et al"""

    def __init__(self, params, lmo):
        self.lmo = lmo
        defaults = dict(lmo=lmo)
        super(StochasticFrankWolfe, self).__init__(params, defaults)


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

                state['step'] += 1
                state['grad_estimate'] += ((1. / (state['step'] + 1)) ** (1/3)
                                           * (grad - state['grad_estimate']))
                update_direction, _ = self.lmo(state['grad_estimate'], p)
                p += (1. / (state['step'] + 1)) * update_direction
        return loss
