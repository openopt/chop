import torch
import numpy as np
from tqdm import tqdm


from chop import utils


class Adversary:
    def __init__(self, method):
        self.method = method

    def perturb(self, data, target, model, criterion,
                step=None, max_iter=20,
                use_best=False,
                initializer=None,
                callback=None,
                *optimizer_args,
                **optimizer_kwargs):
        """Perturbs the batch of datapoints with true label target,
        using specified optimization method.

        Args:
          data: torch.Tensor shape: (batch_size, *)
            batch of datapoints

          target: torch.Tensor shape: (batch_size,)

          model: torch.nn.Module
            model to attack

          step: str or float
            Step size strategy to use for the optimization method.
            If float, value of the step size used.

          max_iter: int
            Maximum number of iterations for the optimization method.

          use_best: bool
            if True, Return best perturbation so far.
            Otherwise, return the last perturbation obtained.

          initializer: callable (optional)
            callable which returns a starting point.
            Typically a random generator on the constraint set. 
            Takes shape as only argument.

          callback: callable (optional)
            called at each iteration of the optimization method.

          *optimizer_args: tuple
            extra arguments for the optimization method

          *optimizer_kwargs: dict
            extra keyword arguments for the optimization method

        Returns:
          adversarial_loss: torch.Tensor of shape (batch_size,)
            vector of losses obtained on the batch

          delta: torch.Tensor of shape (batch_size, *)
            perturbation found"""

        device = data.device
        batch_size = data.size(0)

        @utils.closure
        def loss(delta):
            return -criterion(model(data + delta), target)

        if initializer is None:
            delta0 = torch.zeros_like(data, device=device)

        else:
            delta0 = initializer(data.shape)

        class UseBest:
            def __init__(self):
                self.best = torch.zeros_like(data, device=device)
                self.best_loss = -np.inf * torch.ones(batch_size, device=device)

            def __call__(self, kwargs):
                mask = (-kwargs['fval'] > self.best_loss)
                self.best_loss[mask] = -kwargs['fval'][mask]
                self.best[mask] = kwargs['x'][mask].detach().clone()

                if callback is not None:
                    return callback(kwargs)


        cb = UseBest() if use_best else callback

        sol = self.method(loss, delta0, step=step, max_iter=max_iter, 
                          *optimizer_args, callback=cb,
                          **optimizer_kwargs)

        if use_best:
            return cb.best_loss, cb.best

        return -sol.fval, sol.x

    def attack_dataset(self, loader, model, criterion,
                       step=None, max_iter=20,
                       use_best=False,
                       initializer=None,
                       callback=None,
                       verbose=1,
                       device=None,
                       *optimizer_args,
                       **optimizer_kwargs):

        """Returns a generator of losses, perturbations over
        loader."""

        iterator = enumerate(loader)
        if verbose == 1:
            iterator = tqdm(iterator, total=len(iterator))

        for k, (data, target) in iterator:
            data.to(device)
            target.to(device)

            raise NotImplementedError("The optimization method needs to take "
                                      "arguments which may differ per "
                                      "datapoint.")
            yield self.perturb(data, target, model, criterion, step,
                               max_iter, use_best, initializer, callback,
                               *optimizer_args, **optimizer_kwargs)
