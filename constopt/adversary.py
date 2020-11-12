import torch
import numpy as np

from constopt import utils


class Adversary:
    def __init__(self, method):
        self.method = method

    def perturb(self, data, target, model, criterion,
                step=None, max_iter=20,
                use_best=False,
                random_init=None,
                callback=None,
                *optimizer_args,
                **optimizer_kwargs):

        device = data.device
        batch_size = data.size(0)

        @utils.closure
        def loss(delta):
            return -criterion(model(data + delta), target)

        if random_init is None:
            delta0 = torch.zeros_like(data, device=device)

        else:
            delta0 = random_init(data.shape)

        class UseBest:
            def __init__(self):
                self.best = torch.zeros_like(data)
                self.best_loss = -np.inf * torch.ones(batch_size, device=device)

            def __call__(self, kwargs):
                mask = (kwargs['fval'] < self.best_loss)
                self.best_loss[mask] = kwargs['fval'][mask]
                self.best[mask] = kwargs['x'][mask].detach().clone()

                if callback is not None:
                    return callback(kwargs)


        cb = UseBest() if use_best else callback

        sol = self.method(loss, delta0, step=step, max_iter=max_iter, 
                          *optimizer_args, callback=cb,
                          **optimizer_kwargs)

        if use_best:
            return -cb.best_loss, cb.best

        return sol.fval, sol.x
