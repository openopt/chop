"""
General utility functions.
=========================

"""

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
    if x.is_leaf:
        x.requires_grad_(True)
    else:
        x.retain_grad()
    output = func(x, *args, **kwargs)
    if output.dim() == 0:
        output = output.unsqueeze(0)
    output.backward(torch.ones(batch_size, device=x.device))
    return output.data, x.grad.data


def closure(f):
    @wraps(f)
    def wrapper(x, return_jac=True, *args, **kwargs):
        """Adds jacobian computation when calling function.
        When return_jac is True, returns (value, jacobian)
        instead of just value."""
        if not return_jac:
            val = f(x, *args, **kwargs)
            if val.ndim == 0:
                val = torch.tensor([val], device=val.device)
            return val

        # Reset gradients
        x.grad = None
        return get_func_and_jac(f, x, *args, **kwargs)

    return wrapper


def init_lipschitz(closure, x0, L0=1e-3, n_it=100):
    """Estimates the Lipschitz constant of closure
    for each datapoint in the batch using backtracking line-search.

    Args:
      closure: callable
        returns func_val, jacobian

      x0: torch.tensor of shape (batch_size, *)

      L0: float
        initial guess

      n_it: int
        number of iterations

    Returns:
      Lt: torch.tensor of shape (batch_size,)
    """

    Lt = L0 * torch.ones(x0.size(0), device=x0.device, dtype=x0.dtype)

    f0, grad = closure(x0)
    xt = x0 - bmul((1. / Lt), grad)

    ft = closure(xt, return_jac=False)

    for _ in range(n_it):
        mask = (ft > f0)
        Lt[mask] *= 10.
        xt = x0 - bmul(1. / Lt, grad)
        ft = closure(xt, return_jac=False)

        if not mask.any():
            break
    return Lt


def bdot(tensor, other):
    """Returns the batch-wise dot product between tensor and other.
    Supposes that the shapes are (batch_size, *).
    This includes matrix inner products."""

    t1 = tensor.view(tensor.size(0), -1)
    t2 = other.view(other.size(0), -1)
    return (t1 * t2).sum(dim=-1)


def bmul(tensor, other):
    """Batch multiplies tensor and other"""
    return torch.mul(tensor.T, other.T).T


def bdiv(tensor, other):
    """Batch divides tensor by other"""
    return bmul(tensor, 1. / other)


def bnorm(tensor, *args, **kwargs):
    """Batch vector norms for tensor"""
    batch_size = tensor.size(0)
    return torch.linalg.norm(tensor.reshape(batch_size, -1), dim=-1, *args, **kwargs)

def bmm(tensor, other):
    *batch_dims, m, n = tensor.shape
    *_, n2,  p = other.shape
    if n2 != n:
        raise ValueError(f"Make sure shapes are compatible. Got "
                         f"{tensor.shape}, {other.shape}.")
    t1 = tensor.view(-1, m, n)
    t2 = other.view(-1, n, p)
    return torch.bmm(t1, t2).view(*batch_dims, m, p)


def bmv(tensor, vector):
    return bmm(tensor, vector.unsqueeze(-1)).squeeze(-1)


# TODO: tolerance parameter
def power_iteration(mat, n_iter: int=10, tol: float=1e-6):
    """
    Obtains the largest singular value of a matrix, batch wise,
    and the associated left and right singular vectors.

    Args:
      mat: torch.Tensor of shape (*, M, N)
      n_iter: int
        number of iterations to perform
      tol: float
        Tolerance. Not used for now.
    """
    if n_iter < 1 or type(n_iter) != int:
        raise ValueError("n_iter must be a positive integer.")
    *batch_shapes, m, n = mat.shape
    matT = torch.transpose(mat, -1, -2)
    vec_shape = (*batch_shapes, n)
    # Choose a random vector
    # to decrease the chance that our
    # initial right vector
    # is orthogonal to the first singular vector
    v_k = torch.normal(torch.zeros(vec_shape, device=mat.device),
                       torch.ones(vec_shape, device=mat.device))

    for _ in range(n_iter):
        u_k = bmv(mat, v_k)
        # get singular value
        sigma_k = torch.norm(u_k.view(-1, m), dim=-1).view(*batch_shapes)
        # normalize u
        u_k = bdiv(u_k, sigma_k)

        v_k = bmv(matT, u_k)
        norm_vk = torch.norm(v_k.view(-1, n), dim=-1).view(*batch_shapes)

        # normalize v
        v_k = bmul(v_k, 1. / norm_vk)
    return u_k, sigma_k, v_k
