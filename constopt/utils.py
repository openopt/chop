from functools import wraps
import torch


def closure(f):
    @wraps(f)
    def wrapper(x, return_gradient=True, *args, **kwargs):
        """Adds gradient computation when calling function."""
        if not return_gradient:
            val = f(x, *args, **kwargs)
            return val

        x.grad = None
        val = f(x, *args, **kwargs)
        val.sum().backward()
        return val, x.grad

    return wrapper


def init_lipschitz(closure, x0, L0=1e-3, n_it=100):
    """Estimates the Lipschitz constant of closure
    for each datapoint in the batch using backtracking line-search. x0: torch.tensor of shape (batch_size, *).
    """

    Lt = (torch.ones(x0.size(0)) * L0).to(x0.device)

    f0, grad = closure(x0)
    xt = x0 - bmul((1. / Lt), grad)

    ft = closure(xt, return_gradient=False)

    for _ in range(n_it):
        mask = (ft > f0)
        Lt[mask] *= 10.
        xt = x0 - bmul(1. / Lt, grad)
        ft = closure(xt, return_gradient=False)
    return Lt


def bdot(tensor, other):
    """Returns the batch-wise dot product between tensor and other.
    Supposes that the shapes are (batch_size, *)"""

    t1 = tensor.view(tensor.size(0), -1)
    t2 = other.view(other.size(0), -1)
    return (t1 * t2).sum(dim=-1)


def bmul(tensor, other):
    """Batch multiplies tensor and other"""
    return torch.mul(tensor.T, other.T).T