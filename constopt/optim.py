"""This API is inspired by the COPT project
https://github.com/openopt/copt.

This module contains full gradient optimizers in PyTorch.
These optimizers expect to be called on variables of shape (batch_size, *),
and will perform the optimization point-wise over the batch."""

import torch
import numpy as np
from scipy import optimize
from constopt import utils


def minimize_three_split(
    closure,
    x0,
    prox1=None,
    prox2=None,
    tol=1e-6,
    max_iter=1000,
    verbose=0,
    callback=None,
    line_search=True,
    step_size=None,
    max_iter_backtracking=100,
    backtracking_factor=0.7,
    h_Lipschitz=None,
    *args_prox,
    **kwargs_prox
):

    """Davis-Yin three operator splitting method.
    This algorithm can solve problems of the form

                minimize_x f(x) + g(x) + h(x)

    where f is a smooth function and g and h are (possibly non-smooth)
    functions for which the proximal operator is known.

    Args:
      closure: callable
        Returns the function values and gradient of the objective function.
        With return_gradient=False, returns only the function values.
        Shape of return value: (batch_size, *)

      x0 : torch.Tensor(shape: (batch_size, *))
        Initial guess

      prox1 : callable or None
        prox1(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha.
        alpha can be a scalar or of shape (batch_size).

      prox2 : callable or None
        prox2(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha.
        alpha can be a scalar or of shape (batch_size).

      tol: float
        Tolerance of the stopping criterion.

      max_iter : int
        Maximum number of iterations.

      verbose : int
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

      callback : callable.
        callback function (optional).
        Called with locals() at each step of the algorithm.
        The algorithm will exit if callback returns False.

      line_search : boolean
        Whether to perform line-search to estimate the step sizes.

      step_size : float or tensor(shape: (batch_size,)) or None
        Starting value(s) for the line-search procedure.
        if None, step_size will be estimated for each datapoint in the batch.

      max_iter_backtracking: int
        maximun number of backtracking iterations.  Used in line search.

      backtracking_factor: float
        the amount to backtrack by during line search.

      args_prox: iterable
        (optional) Extra arguments passed to the prox functions.

      kwargs_prox: dict
        (optional) Extra keyword arguments passed to the prox functions.


    Returns:
      res : OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution tensor, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.
    """

    success = torch.zeros(x0.size(0), dtype=bool)
    if not max_iter_backtracking > 0:
        raise ValueError("Line search iterations need to be greater than 0")

    LS_EPS = np.finfo(np.float).eps

    if prox1 is None:

        def prox1(x, s=None, *args):
            return x

    if prox2 is None:

        def prox2(x, s=None, *args):
            return x

    x = x0.detach().clone().requires_grad_(True)

    if step_size is None:
        line_search = True
        step_size = 1.0 / utils.init_lipschitz(closure, x)

    with torch.no_grad():
        z = prox2(x, step_size, *args_prox)
        z.requires_grad = True

    fk, grad_fk = closure(z)

    with torch.no_grad():
        x = prox1(z - utils.bmul(step_size, grad_fk), step_size, *args_prox)
        u = torch.zeros_like(x)

    for it in range(max_iter):
        print(it)
        fk, grad_fk = closure(z)
        with torch.no_grad():
            x = prox1(z - utils.bmul(step_size, u + grad_fk), step_size, *args_prox)
            incr = x - z
            norm_incr = torch.norm(incr.view(incr.size(0), -1), dim=-1)
            rhs = fk + utils.bdot(grad_fk, incr) + ((norm_incr ** 2) / (2 * step_size))
            ls_tol = closure(x, return_gradient=False)
            mask = torch.bitwise_and(norm_incr > 1e-7, line_search)
            ls = mask.any()
            # TODO: optimize code in this loop using mask
            for it_ls in range(max_iter_backtracking):
                rhs[mask] = fk[mask] + utils.bdot(grad_fk[mask], incr[mask]) + ((norm_incr ** 2) / (2 * step_size[mask]))
                ls_tol[mask] = closure(x, return_gradient=False)[mask] - rhs[mask]
                mask &= (ls_tol <= LS_EPS)
                step_size[mask] *= backtracking_factor

            z = prox2(x + utils.bmul(step_size, u), step_size, *args_prox).requires_grad_(True)
            u += utils.bmul(x - z, 1. / step_size)
            certificate = norm_incr / step_size

            if ls and h_Lipschitz is not None:
                if h_Lipschitz == 0:
                    step_size = step_size * 1.02
                else:
                    quot = h_Lipschitz ** 2
                    tmp = torch.sqrt(step_size ** 2 + (2 * step_size / quot) * (-ls_tol))
                    step_size = torch.min(tmp, step_size * 1.02)

        if callback is not None:
            if callback(locals()) is False:
                break

        success = torch.bitwise_and(certificate < tol, it > 0)
        if success.all():
            break

    return optimize.OptimizeResult(x=x, success=success, nit=it)


def minimize_pgd_madry(closure, x0, prox, lmo, step_size=None, max_iter=200, prox_args=(), callback=None):
    x = x0.detach().clone()

    for it in range(max_iter):
        x.requires_grad = True
        _, grad = closure(x)
        with torch.no_grad():
            update_direction, _ = lmo(-grad, x)
            update_direction += x
            x = prox(x + step_size * update_direction, step_size, *prox_args)

        if callback is not None:
            if callback(locals()) is False:
                break

    return optimize.OptimizeResult(x=x, nit=it)


def minimize_pgd(closure, x0, prox, step_size=None, max_iter=200,
                 *prox_args,
                 callback=None):
    """
    Performs Projected Gradient Descent on batch of objectives of form:
      f(x) + g(x).
    We suppose we have access to gradient computation for f through closure,
    and to the proximal operator of g in prox.

    Args:
      closure: callable

      x0: torch.Tensor of shape (batch_size, *).

      prox: callable
        proximal operator of g

      step_size: None or float or torch.tensor of shape (batch_size,).
        step size to be used. If None, will be estimated at the beginning using line search.

      max_iter: int
        number of iterations to perform.

      prox_args: tuple
        (optional) additional args for prox

      callback: callable
        (optional) Any callable called on locals() at the end of each iteration.
        Often used for logging.
    """
    x = x0.detach().clone()

    if step_size is None:
        # estimate lipschitz constant
        L = utils.init_lipschitz(closure, x0)
        step_size = 1. / L

    if type(step_size) == float:
        step_size = torch.ones(x0.size(0)) * step_size


    for it in range(max_iter):
        x.requires_grad = True
        _, grad = closure(x)
        with torch.no_grad():
            x = prox(x - utils.bmul(step_size, grad), step_size, *prox_args)

        if callback is not None:
            if callback(locals()) is False:
                break

    fval, grad = closure(x)
    return optimize.OptimizeResult(x=x, nit=it, fval=fval, grad=grad)
