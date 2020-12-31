"""
Full-gradient optimizers.
=========================

This module contains full gradient optimizers in PyTorch.
These optimizers expect to be called on variables of shape (batch_size, *),
and will perform the optimization point-wise over the batch.

This API is inspired by the COPT project
https://github.com/openopt/copt.
"""

from numbers import Number
import warnings
import torch
import numpy as np
from scipy import optimize
from numbers import Number
from chop import utils


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
  step=None,
  max_iter_backtracking=100,
  backtracking_factor=0.7,
  h_Lipschitz=None,
  *args_prox
  ):

    """Davis-Yin three operator splitting method.
    This algorithm can solve problems of the form

                minimize_x f(x) + g(x) + h(x)

    where f is a smooth function and g and h are (possibly non-smooth)
    functions for which the proximal operator is known.

    Remark: this method returns x = prox1(...). If g and h are two indicator
      functions, this method only garantees that x is feasible for the first.
      Therefore if one of the constraints is a hard constraint,
      make sure to pass it to prox1.

    Args:
      closure: callable
        Returns the function values and gradient of the objective function.
        With return_gradient=False, returns only the function values.
        Shape of return value: (batch_size, *)

      x0 : torch.Tensor(shape: (batch_size, *))
        Initial guess

      prox1 : callable or None
        prox1(x, step_size, *args) returns the proximal operator of g at xa
        with parameter step_size.
        step_size can be a scalar or of shape (batch_size,).

      prox2 : callable or None
        prox2(x, step_size, *args) returns the proximal operator of g at xa
        with parameter step_size.
        alpha can be a scalar or of shape (batch_size,).

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

        @torch.no_grad()
        def prox1(x, s=None, *args):
            return x

    if prox2 is None:
        @torch.no_grad()
        def prox2(x, s=None, *args):
            return x

    x = x0.detach().clone().requires_grad_(True)
    batch_size = x.size(0)

    if step is None:
        line_search = True
        step_size = 1.0 / utils.init_lipschitz(closure, x)

    elif isinstance(step, Number):
        step_size = step * torch.ones(batch_size,
                                      device=x.device,
                                      dtype=x.dtype)

    else:
        raise ValueError("step must be float or None.")

    z = prox2(x, step_size, *args_prox)
    z = z.clone().detach()
    z.requires_grad_(True)

    fval, grad = closure(z)

    x = prox1(z - utils.bmul(step_size, grad), step_size, *args_prox)
    u = torch.zeros_like(x)

    for it in range(max_iter):
        z.requires_grad_(True)
        fval, grad = closure(z)
        with torch.no_grad():
            x = prox1(z - utils.bmul(step_size, u + grad), step_size, *args_prox)
            incr = x - z
            norm_incr = torch.norm(incr.view(incr.size(0), -1), dim=-1)
            rhs = fval + utils.bdot(grad, incr) + ((norm_incr ** 2) / (2 * step_size))
            ls_tol = closure(x, return_jac=False)
            mask = torch.bitwise_and(norm_incr > 1e-7, line_search)
            ls = mask.detach().clone()
            # TODO: optimize code in this loop using mask
            for it_ls in range(max_iter_backtracking):
                if not(mask.any()):
                    break
                rhs[mask] = fval[mask] + utils.bdot(grad[mask], incr[mask])
                rhs[mask] += utils.bmul(norm_incr[mask] ** 2, 1. / (2 * step_size[mask]))

                ls_tol[mask] = closure(x, return_jac=False)[mask] - rhs[mask]
                mask &= (ls_tol > LS_EPS)
                step_size[mask] *= backtracking_factor

            z = prox2(x + utils.bmul(step_size, u), step_size, *args_prox)
            u += utils.bmul(x - z, 1. / step_size)
            certificate = utils.bmul(norm_incr, 1. / step_size)

        if callback is not None:
            if callback(locals()) is False:
                break

        success = torch.bitwise_and(certificate < tol, it > 0)
        if success.all():
            break

    return optimize.OptimizeResult(x=x, success=success, nit=it, fval=fval, certificate=certificate)


def minimize_pgd_madry(closure, x0, prox, lmo, step=None, max_iter=200, prox_args=(), callback=None):
    x = x0.detach().clone()
    batch_size = x.size(0)

    if step is None:
        # estimate lipschitz constant
        # TODO: this is not the optimal step-size (if there even is one.)
        # I don't recommend to use this.
        L = utils.init_lipschitz(closure, x0)
        step_size = 1. / L

    elif isinstance(step, Number):
        step_size = torch.ones(batch_size, device=x.device) * step

    elif isinstance(step, torch.Tensor):
        step_size = step

    else:
        raise ValueError(f"step must be a number or a torch Tensor, got {step} instead")

    for it in range(max_iter):
        x.requires_grad = True
        _, grad = closure(x)
        with torch.no_grad():
            update_direction, _ = lmo(-grad, x)
            update_direction += x
            x = prox(x + utils.bmul(step_size, update_direction),
                     step_size, *prox_args)

        if callback is not None:
            if callback(locals()) is False:
                break

    fval, grad = closure(x)
    return optimize.OptimizeResult(x=x, nit=it, fval=fval, grad=grad)


def backtracking_pgd(closure, prox, step_size, x, grad, increase=1.01, decrease=.6, max_iter_backtracking=1000):

    batch_size = x.size(0)
    rhs = -np.inf * torch.ones(batch_size)
    lhs = np.inf * torch.ones(batch_size)

    need_to_backtrack = lhs > rhs

    while (~need_to_backtrack).any():
        step_size[~need_to_backtrack] *= increase

    while need_to_backtrack.any():
      with torch.no_grad():
          x_candidate = prox(x - utils.bmul(step_size, grad), step_size)

      lhs = closure(x_candidate, return_jac=False)
      rhs = (closure(x, return_jac=False) - utils.bdot(grad, x - x_candidate)
              + utils.bmul(1. / (2 * step_size),
                          torch.norm((x - x_candidate).view(x.size(0), -1),
                                      dim=-1))) ** 2


def minimize_pgd(closure, x0, prox, step='backtracking', max_iter=200,
                 max_iter_backtracking=1000,
                 backtracking_factor=.6,
                 tol=1e-8,
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

      step: 'backtracking' or float or torch.tensor of shape (batch_size,) or None.
        step size to be used. If None, will be estimated at the beginning
        using line search.
        If 'backtracking', will be estimated at each step using backtracking line search.

      max_iter: int
        number of iterations to perform.

      max_iter_backtracking: int
        max number of iterations in the backtracking line search

      backtracking_factor: float
        factor by which to multiply the step sizes during line search

      tol: float
        stops the algorithm when the certificate is smaller than tol
        for all datapoints in the batch

      prox_args: tuple
        (optional) additional args for prox

      callback: callable
        (optional) Any callable called on locals() at the end of each iteration.
        Often used for logging.
    """
    x = x0.detach().clone()
    batch_size = x.size(0)

    if step is None:
        # estimate lipschitz constant
        L = utils.init_lipschitz(closure, x0)
        step_size = 1. / L

    elif step == 'backtracking':
        L = 1.8 * utils.init_lipschitz(closure, x0)
        step_size = 1. / L

    elif type(step) == float:
        step_size = step * torch.ones(batch_size, device=x.device)

    else:
        raise ValueError("step must be float or backtracking or None")

    for it in range(max_iter):
        x.requires_grad = True

        fval, grad = closure(x)

        x_next = prox(x - utils.bmul(step_size, grad), step_size, *prox_args)
        update_direction = x_next - x

        if step == 'backtracking':
            step_size *= 1.1
            mask = torch.ones(batch_size, dtype=bool, device=x.device)

            with torch.no_grad():
                for _ in range(max_iter_backtracking):
                    f_next = closure(x_next, return_jac=False)
                    rhs = (fval
                           + utils.bdot(grad, update_direction)
                           + utils.bmul(utils.bdot(update_direction,
                                                   update_direction),
                                        1. / (2. * step_size))
                           )
                    mask = f_next > rhs

                    if not mask.any():
                        break

                    step_size[mask] *= backtracking_factor
                    x_next = prox(x - utils.bmul(step_size,
                                                 grad),
                                  step_size[mask],
                                  *prox_args)
                    update_direction[mask] = x_next[mask] - x[mask]
                else:
                    warnings.warn("Maximum number of line-search iterations "
                                  "reached.")

        with torch.no_grad():
            cert = torch.norm(utils.bmul(update_direction, 1. / step_size),
                              dim=-1)
            x.copy_(x_next)
            if (cert < tol).all():
                break

        if callback is not None:
            if callback(locals()) is False:
                break

    fval, grad = closure(x)
    return optimize.OptimizeResult(x=x, nit=it, fval=fval, grad=grad,
                                   certificate=cert)


def minimize_frank_wolfe(closure, x0, lmo, step='sublinear',
                         max_iter=200, callback=None, *args, **kwargs):
    """Performs the Frank-Wolfe algorithm on a batch of objectives of the form
      min_x f(x)
      s.t. x in C

    where we have access to the Linear Minimization Oracle (LMO) of the constraint set C,
    and the gradient of f through closure.

    Args:
      closure: callable
        gives function values and the jacobian of f.

      x0: torch.Tensor of shape (batch_size, *).
        initial guess

      lmo: callable
        Returns update_direction, max_step_size

      step: float or 'sublinear'
        step-size scheme to be used.

      max_iter: int
        max number of iterations.

      callback: callable
        (optional) Any callable called on locals() at the end of each iteration.
        Often used for logging.
    """
    x = x0.detach().clone()
    batch_size = x.size(0)
    if not (isinstance(step, Number) or step == 'sublinear'):
        raise ValueError(f"step must be a float or 'sublinear', got {step} instead.")

    if isinstance(step, Number):
        step_size = step * torch.ones(batch_size, device=x.device, dtype=x.dtype)

    cert = np.inf * torch.ones(batch_size, device=x.device)

    for it in range(max_iter):

        x.requires_grad = True
        fval, grad = closure(x)
        update_direction, max_step_size = lmo(-grad, x)
        cert = utils.bdot(-grad, update_direction)

        if step == 'sublinear':
            step_size = 2. / (it + 2) * torch.ones(batch_size, dtype=x.dtype, device=x.device)

        with torch.no_grad():
            step_size = torch.min(step_size, max_step_size)
            x += utils.bmul(update_direction, step_size)

        if callback is not None:
            if callback(locals()) is False:
                break

    fval, grad = closure(x)
    return optimize.OptimizeResult(x=x, nit=it, fval=fval, grad=grad,
                                   certificate=cert)
