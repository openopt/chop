"""This API is inspired by the COPT project
https://github.com/openopt/copt.

This module contains full gradient optimizers in PyTorch."""

import torch
import numpy as np
from constopt import opt_utils


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
    args_prox=(),
):

    """Davis-Yin three operator splitting method.
    This algorithm can solve problems of the form

                minimize_x f(x) + g(x) + h(x)

    where f is a smooth function and g and h are (possibly non-smooth)
    functions for which the proximal operator is known.

    Args:
      f_grad: callable
        Returns the function value and gradient of the objective function.
        With return_gradient=False, returns only the function value.

      x0 : array-like
        Initial guess

      prox_1 : callable or None
        prox_1(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha.

      prox_2 : callable or None
        prox_2(x, alpha, *args) returns the proximal operator of g at xa
        with parameter alpha.

      tol: float
        Tolerance of the stopping criterion.

      max_iter : int
        Maximum number of iterations.

      verbose : int
        Verbosity level, from 0 (no output) to 2 (output on each iteration)

      callback : callable.
        callback function (optional). Takes a single argument (x) with the
        current coefficients in the algorithm. The algorithm will exit if
        callback returns False.

      line_search : boolean
        Whether to perform line-search to estimate the step size.

      step_size : float
        Starting value for the line-search procedure.

      max_iter_backtracking: int
        maximun number of backtracking iterations.  Used in line search.

      backtracking_factor: float
        the amount to backtrack by during line search.

      args_prox: tuple
        optional Extra arguments passed to the prox functions


    Returns:
      res : OptimizeResult
        The optimization result represented as a
        ``scipy.optimize.OptimizeResult`` object. Important attributes are:
        ``x`` the solution array, ``success`` a Boolean flag indicating if
        the optimizer exited successfully and ``message`` which describes
        the cause of the termination. See `scipy.optimize.OptimizeResult`
        for a description of other attributes.
    """

    success = False
    if not max_iter_backtracking > 0:
        raise ValueError("Line search iterations need to be greater than 0")

    if prox1 is None:

        def prox1(x, s=None, *args):
            return x

    if prox2 is None:

        def prox2(x, s=None, *args):
            return x

    if step_size is None:
        line_search = True
        step_size = 1.0 / opt_utils.init_lipschitz(closure, x0)

    z = prox2(x0, step_size, *args_prox)
    LS_EPS = np.finfo(np.float).eps

    fk, grad_fk = closure(z)

    fk = closure(z, return_gradient=False)
    fk.backward()
    grad_fk = z.grad

    x = prox1(z - step_size * grad_fk, step_size, *args_prox)
    u = torch.zeros_like(x)

    for it in range(max_iter):

        fk, grad_fk = closure(z)
        x = prox1(z - step_size * (u + grad_fk), step_size, *args_prox)
        incr = x - z
        norm_incr = np.linalg.norm(incr)
        ls = norm_incr > 1e-7 and line_search
        if ls:
            for it_ls in range(max_iter_backtracking):
                rhs = fk + grad_fk.dot(incr) + (norm_incr ** 2) / (2 * step_size)
                ls_tol = closure(x, return_gradient=False) - rhs
                if ls_tol <= LS_EPS:
                    # step size found
                    # if ls_tol > 0:
                    #     ls_tol = 0.
                    break
                else:
                    step_size *= backtracking_factor

        z = prox2(x + step_size * u, step_size, *args_prox)
        u += (x - z) / step_size
        certificate = norm_incr / step_size

        if ls and h_Lipschitz is not None:
            if h_Lipschitz == 0:
                step_size = step_size * 1.02
            else:
                quot = h_Lipschitz ** 2
                tmp = np.sqrt(step_size ** 2 + (2 * step_size / quot) * (-ls_tol))
                step_size = min(tmp, step_size * 1.02)

        if callback is not None:
            if callback(locals()) is False:
                break

        if it > 0 and certificate < tol:
            success = True
            break

    return x


def minimize_pgd_madry(x0, f_grad, prox, lmo, step_size=None, max_iter=200, prox_args=(), callback=None):
    x = x0.detach().clone()

    for it in range(max_iter):
        x.requires_grad = True
        loss, grad = f_grad(x)
        with torch.no_grad():
            update_direction, _ = lmo(-grad, x)
            update_direction += x
            x = prox(x + step_size * update_direction, step_size, *prox_args)

        if callback:
            callback(locals())
            
    return x