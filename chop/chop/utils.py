"""
General utility functions
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
    x.requires_grad = True
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
    Supposes that the shapes are (batch_size, *)"""

    t1 = tensor.view(tensor.size(0), -1)
    t2 = other.view(other.size(0), -1)
    return (t1 * t2).sum(dim=-1)


def bmul(tensor, other):
    """Batch multiplies tensor and other"""
    return torch.mul(tensor.T, other.T).T


def bmm(tensor, other):
    t1 = tensor.view(-1, tensor.size(-2), tensor.size(-1))
    t2 = other.view(-1, other.size(-2), other.size(-1))
    return torch.bmm(t1, t2).view(-1, tensor.size(-2), other.size(-1))


# TODO: tolerance parameter
def power_iteration(mat, n_iter: int, tol: float):
    """
    Obtains the largest singular value of a matrix, batch wise,
    and the associated left and right singular vectors.

    Args:
      mat: torch.Tensor of shape (*, M, N)
      n_iter: int
        number of iterations to perform
      tol: float
        Tolerance
    """
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    x_k = torch.normal(torch.zeros_like(mat), 1., device=mat.device)

    for _ in range(n_iter):
        # calculate the matrix-by-vector product Ab
        x_k = torch.torch.matmul(mat.T, mat)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k