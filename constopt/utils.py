from functools import wraps
import torch


def get_func_and_jac(func, x, *args, **kwargs):
    """Computes the jacobian of a batch-wise separable function func of x.
    func returns a torch.Tensor of shape (batch_size,) when
    x is a torch.Tensor of shape (batch_size, *).
    Adapted from
    https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa 
    by Shane Baratt"""

    batch_size = x.size(0)
    x.requires_grad = True
    output = func(x, *args, **kwargs)
    output.backward(torch.ones(batch_size))
    return output.data, x.grad.data


def closure(f):
    @wraps(f)
    def wrapper(x, return_jac=True, *args, **kwargs):
        """Adds jacobian computation when calling function.
        When return_jac is True, returns (value, jacobian)
        instead of just value."""
        if not return_jac:
            val = f(x, *args, **kwargs)
            return val

        x.grad = None
        return get_func_and_jac(f, x, *args, **kwargs)

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