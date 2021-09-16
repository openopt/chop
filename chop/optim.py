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
from collections import defaultdict

import torch

import numpy as np

from scipy import optimize
from chop import utils
from chop import constraints


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


def minimize_pgd(closure, x0, prox=None, step='backtracking', max_iter=200,
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

    if prox is None:
        def prox(x, s=None):
            return x

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

    Returns:
      
      result: optimize.OptimizeResult object
        Holds the result of the optimization, and certificates of convergence.
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


def update_active_set(active_set,
                      fw_idx, away_idx,
                      step_size):

    max_step_size = active_set[away_idx]
    active_set[fw_idx] += step_size
    active_set[away_idx] -= step_size
    
    if active_set[away_idx] == 0.:
        # drop step: remove vertex from active set
        del active_set[away_idx]
    if active_set[away_idx] < 0.:
        raise ValueError(f"The step size used is too large. "
                         f"{step_size: .3f} vs. {max_step_size:.3f}")

    return active_set


def backtracking_fw(
    x,
    fval,
    old_fval,
    closure,
    certificate,
    lipschitz_t,
    max_step_size,
    update_direction,
    norm_update_direction,
    tol=torch.finfo(torch.float32).eps
    ):
    """Performs backtracking line search for Frank-Wolfe algorithms."""

    ratio_decrease = .9
    ratio_increase = 2.
    max_linesearch_iter = 100

    if old_fval is not None:
        tmp = (certificate ** 2) / (2 * (old_fval - fval) * norm_update_direction)
        lipschitz_t = max(min(tmp, lipschitz_t), lipschitz_t * ratio_decrease)

    for _ in range(max_linesearch_iter):
        step_size_t = certificate / (norm_update_direction * lipschitz_t)
        if step_size_t < max_step_size:
            rhs = -0.5 * step_size_t * certificate
        else:
            step_size_t = max_step_size
            rhs = (
                -step_size_t * certificate
                + 0.5 * (step_size_t ** 2) * lipschitz_t * norm_update_direction
            )
        fval_next, grad_next = closure(x + step_size_t * update_direction)
        if fval_next - fval <= rhs + tol:
            # .. sufficient decrease condition verified ..
            break
        else:
            lipschitz_t *= ratio_increase
    else:
        warnings.warn(
            "Exhausted line search iterations in minimize_frank_wolfe", RuntimeWarning
        )
    return step_size_t, lipschitz_t, fval_next, grad_next


def minimize_pairwise_frank_wolfe(
    closure,
    x0_idx,
    polytope,
    step='backtracking',
    lipschitz=None,
    max_iter=200,
    tol=1e-6,
    callback=None
    ):
    """Minimize using Pairwise Frank-Wolfe.
    
    WARNING: This implementation is different from other functions in this file.
    As of now, it does not handle batched problems, and only handles Polytope constraints.

    Args:
      closure: closure of the function to minimize
      x0_idx: the starting vertex on the polytope
      polytope: the polytope constraint
      step: backtracking line search as defined in [1]
      lipschitz: an initial Lipschitz estimate of the gradient
      max_iter: maximum number of iterations
      tol: tolerance on the Frank-Wolfe gap
      callback: a callable callback function
    """

    if not isinstance(polytope, constraints.Polytope):
      raise ValueError("polytope must be a `chop.constraints.Polytope`.")

    x = polytope.vertices[x0_idx].detach().clone()
    x = x.unsqueeze(0)
    active_set = defaultdict(float)
    active_set[x0_idx] = 1.

    cert = float('inf')

    x.requires_grad = True
    fval, grad = closure(x)
    old_fval = None


    lipschitz_t = None
    step_size = None

    if lipschitz is not None:
        lipschitz_t = lipschitz 

    for it in range(max_iter):
        update_direction, fw_idx, away_idx, max_step_size = polytope.lmo_pairwise(-grad, x, active_set)
        norm_update_direction = torch.linalg.norm(update_direction) ** 2
        cert = utils.bdot(update_direction, -grad)
        
        if lipschitz_t is None:
            eps = 1e-3
            grad_eps = closure(x + eps * update_direction)[1]
            lipschitz_t = torch.linalg.norm(grad-grad_eps) / (
                eps * torch.sqrt(norm_update_direction)
            )
            print(f"Estimated L_t = {lipschitz_t}")

        if cert <= tol:
            break

        if step == 'DR':
            step_size = min(
                cert / (norm_update_direction * lipschitz_t), max_step_size
            )
            fval_next, grad_next = closure(x + step_size * update_direction)
        elif step == 'backtracking':
            step_size, lipschitz_t, fval_next, grad_next = backtracking_fw(
                x, fval, old_fval, closure, cert, lipschitz_t, max_step_size,
                update_direction, norm_update_direction
            )
            
        elif step == 'sublinear':
            step_size = 2. / (it + 2.)
            step_size = min(step_size, max_step_size)
            fval_next, grad_next = closure(x + step_size * update_direction)
            
        if callback is not None:
            if callback(locals()) is False:  # pylint: disable=g-bool-id-comparison
                break

        with torch.no_grad():
            x.add_(step_size * update_direction)

        update_active_set(
            active_set,
            fw_idx,
            away_idx,
            step_size)

        old_fval = fval
        fval, grad = fval_next, grad_next
    if callback is not None:
        callback(locals())

    return optimize.OptimizeResult(x=x.data, nit=it, certificate=cert, active_set=active_set,
                                   fval=fval, grad=grad)


def minimize_alternating_fw_prox(closure, x0, y0, prox=None, lmo=None, lipschitz=1e-3,
                                 step='sublinear', line_search=None, max_iter=200, callback=None,
                                 *args, **kwargs):
    """
    Implements algorithm from [Garber et al. 2018]
    https://arxiv.org/abs/1802.05581

    to solve the following problem


    ..math::
        \min_{x, y} f(x + y) + R_x(x) + R_y(y).

    We suppose that $f$ is $L$-smooth and that
    we have access to the following operators:
    
      - a generalized LMO for $R_y$: 
    ..math::
        gLMO(w) = \text{argmin}_w R_y(w) + \langle w, \nabla f(x_t + y_t) \rangle

      - a prox operator for $R_x$:
    ..math::
        prox(v) = \text{argmin}_v R_x(v) + \langle v, \nabla f(x_t+ y_t) \rangle + \frac{\gamma_t L}{2} \|v + w_t - (x_t + y_t)\|^2
    
    Args:
      x0: torch.Tensor of shape (batch_size, *)
        starting point for x

      y0: torch.Tensor of shape (batch_size, *)
        starting point for y
      
      prox: function
        proximal operator for R_x

      lmo: function
        generalized LMO operator for R_y. If R_y is an indicator function,
        it reduces to the usual LMO operator.

      lipschitz: float
        initial guess of the lipschitz constant of f

      step: float or 'sublinear'
        step-size scheme to be used.

      max_iter: int
        max number of iterations.

      callback: callable
        (optional) Any callable called on locals() at the end of each iteration.
        Often used for logging.
    
    Returns:
      
      result: optimize.OptimizeResult object
        Holds the result of the optimization, and certificates of convergence.
    """

    x = x0.detach().clone()
    y = y0.detach().clone()
    batch_size = x.size(0)

    if x.shape != y.shape:
        raise ValueError(f"x, y should have the same shape. Got {x.shape}, {y.shape}.")

    if not (isinstance(step, Number) or step == 'sublinear'):
        raise ValueError(f"step must be a float or 'sublinear', got {step} instead.")

    if isinstance(step, Number):
        step_size = step * torch.ones(batch_size, device=x.device, dtype=x.dtype)

    # TODO: add error catching for L0
    Lt = lipschitz

    for it in range(max_iter):

        if step == 'sublinear':
            step_size = 2. / (it + 2) * torch.ones(batch_size, device=x.device)

        x.requires_grad_(True)
        y.requires_grad_(True)
        z = x + y

        f_val, grad = closure(z)

        # estimate Lipschitz constant with backtracking line search
        Lt = utils.init_lipschitz(closure, z, L0=Lt)

        y_update, max_step_size = lmo(-grad, y)
        with torch.no_grad():
            w = y_update + y
        prox_step_size = utils.bmul(step_size, Lt)
        v = prox(z - w - utils.bdiv(grad, prox_step_size), prox_step_size)

        with torch.no_grad():
            if line_search is None:
                step_size = torch.min(step_size, max_step_size)
            else:
                step_size = line_search(locals())

            y += utils.bmul(step_size, y_update)
            x_update = v - x
            x += utils.bmul(step_size, x_update)

        if callback is not None:
            if callback(locals()) is False:
                break

    fval, grad = closure(x + y)
    # TODO: add a certificate of optimality
    result = optimize.OptimizeResult(x=x, y=y, nit=it, fval=fval, grad=grad, certificate=None)
    return result
