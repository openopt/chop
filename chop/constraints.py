"""
Constraints.
===========
This module contains classes representing constraints.
The methods on each constraint object function batch-wise.
Reshaping will be of order if the constraints are used on the parameters of a model.
This uses an API similar to the one for
the COPT project, https://github.com/openopt/copt.
Part of this code is adapted from https://github.com/ZIB-IOL."""

from copy import deepcopy
from collections import defaultdict
import warnings

import torch

import numpy as np
from scipy.stats import expon
from torch.distributions import Laplace, Normal
from chop import utils


@torch.no_grad()
def get_avg_init_norm(layer, param_type=None, ord=2, repetitions=100):
    """Computes the average norm of default layer initialization"""
    output = 0
    for _ in range(repetitions):
        layer.reset_parameters()
        warnings.warn("torch.norm is deprecated. Think about updating this.")
        output += torch.norm(getattr(layer, param_type), p=ord).item()
    return float(output) / repetitions


def is_bias(name, param):
    return ('bias' in name) or (param.ndim < 2)


@torch.no_grad()
def make_model_constraints(model, ord=2, value=300, mode='initialization', constrain_bias=False):
    """Create Ball constraints for each layer of model. Ball radius depends on mode (either radius or
    factor to multiply average initialization norm with)"""
    constraints = []

    # Compute average init norms if necessary
    init_norms = dict()

    if (ord == 'nuc') and constrain_bias:
        msg = "'nuc' constraints cannot constrain bias."
        warnings.warn(msg)
        constrain_bias = False

    if mode == 'initialization':
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if (hasattr(layer, entry) and
                                                                             type(getattr(layer, entry)) != type(
                            None))]:
                    param = getattr(layer, param_type)
                    shape = param.shape
                    # TODO: figure out how to set the constraint size for NuclearNormBall constraint
                    avg_norm = get_avg_init_norm(layer, param_type=param_type, ord=2)
                    if avg_norm == 0.0:
                        # Catch unlikely case that weight/bias is 0-initialized (e.g. BatchNorm does this)
                        avg_norm = 1.0
                    init_norms[shape] = avg_norm

    for name, param in model.named_parameters():
        if is_bias(name, param):
            constraint = None
        else:
            print(name)
            if mode == 'radius':
                alpha = value
            elif mode == 'initialization':
                alpha = value * init_norms[param.shape]
            else:
                msg = f"Unknown mode {mode}."
                raise ValueError(msg)
            if (type(ord) == int) or (ord == np.inf):
                constraint = make_LpBall(alpha, p=ord)
            elif ord == 'nuc':
                constraint = NuclearNormBall(alpha)
            else:
                msg = f"ord {ord} is not supported."
                raise ValueError(msg)
        constraints.append(constraint)
    return constraints


@torch.no_grad()
def make_feasible(model, proxes):
    """
    Projects all parameters of model onto the associated constraint set,
    using its prox operator (really a projection here).
    This function operates in-place.

    Args:
      model: torch.nn.Module
        Model to make feasible
      prox: [callable]
        List of projection operators
    """
    for param, prox in zip(model.parameters(), proxes):
        if prox is not None:
            param.copy_(prox(param.unsqueeze(0)).squeeze(0))

@torch.no_grad()
def euclidean_proj_simplex(v, s=1.):
    r""" Compute the Euclidean projection on a positive simplex
  Solves the optimization problem (using the algorithm from [1]):
    ..math::
      min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
  Parameters
  ----------
  v: (n,) numpy array,
      n-dimensional vector to project
  s: float, optional, default: 1,
      radius of the simplex
  Returns
  -------
  w: (n,) numpy array,
      Euclidean projection of v on the simplex
  Notes
  -----
  The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
  Better alternatives exist for high-dimensional sparse vectors (cf. [1])
  However, this implementation still easily scales to millions of dimensions.
  References
  ----------
  [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
      John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
      International Conference on Machine Learning (ICML 2008)
      http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
  """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=-1)
    # get the number of > 0 components of the optimal solution
    rho = (u * torch.arange(1, n + 1, device=v.device) > (cssv - s)).sum() - 1
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = torch.clamp(v - theta, min=0)
    return w


@torch.no_grad()
def euclidean_proj_l1ball(v, s=1.):
    """ Compute the Euclidean projection on a L1-ball
  Solves the optimization problem (using the algorithm from [1]):
    ..math::
      min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
      
  Args:
  
    v: (n,) numpy array,
      n-dimensional vector to project
    s: float, optional, default: 1,
      radius of the L1-ball
      
  Returns:
    w: (n,) numpy array,
      Euclidean projection of v on the L1-ball of radius s
  Notes
  -----
  Solves the problem by a reduction to the positive simplex case
  See also
  --------
  euclidean_proj_simplex
  """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    if len(v.shape) > 1:
        raise ValueError
    # compute the vector of absolute values
    u = abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= torch.sign(v)
    return w


class LpBall:
    def __init__(self, alpha):
        if not 0. <= alpha:
            raise ValueError("Invalid constraint size alpha: {}".format(alpha))
        self.alpha = alpha
        self.active_set = defaultdict(float)

    @torch.no_grad()
    def fw_gap(self, grad, iterate):
        update_direction, _ = self.lmo(-grad, iterate)
        return utils.bdot(-grad, update_direction)

    @torch.no_grad()
    def random_point(self, shape):
        """
        Sample uniformly from the constraint set.
        L1 and L2 are implemented here.
        Linf implemented in the subclass.
        https://arxiv.org/abs/math/0503650
        """
        if self.p == 2:
            distrib = Normal(0, 1)
        elif self.p == 1:
            distrib = Laplace(0, 1)
        x = distrib.sample(shape)
        e = expon(.5).rvs()
        denom = torch.sqrt(e + (x ** 2).sum())
        return self.alpha * x / denom

    def __mul__(self, other):
        """Scales the constraint by a scalar"""
        ret = deepcopy(self)
        ret.alpha *= other
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self.alpha *= other
        return self

    def __truediv__(self, other):
        ret = deepcopy(self)
        ret.alpha /= other
        return ret

    @torch.no_grad()
    def make_feasible(self, model):
        """Projects all parameters of model into the constraint set."""

        for idx, (name, param) in enumerate(model.named_parameters()):
            param.copy_(self.prox(param))

    def is_feasible(self, x, rtol=1e-5, atol=1e-7):
        """Checks if x is a feasible point of the constraint."""
        p_norms = (x ** self.p).reshape(x.size(0), -1).sum(-1)
        return p_norms.pow(1. / self.p) <= self.alpha * (1. + rtol) + atol


class LinfBall(LpBall):
    p = np.inf

    @torch.no_grad()
    def prox(self, x, step_size=None):
        """Projection onto the L-infinity ball.

        Args:
          x: torch.Tensor of shape (batchs_size, *)
            tensor to project
          step_size: Any
            Not used here
        
        Returns:
          p: torch.Tensor, same shape as x
            projection of x onto the L-infinity ball.
        """
        if torch.max(abs(x)) <= self.alpha:
            return x
        return torch.clamp(x, min=-self.alpha, max=self.alpha)

    @torch.no_grad()
    def lmo(self, grad, iterate):
        """Linear Maximization Oracle.
        Return s - iterate with s solving the linear problem

        ..math::
            max_{||s||_\infty <= alpha} <grad, s>

        Args:
          grad: torch.Tensor of shape (batch_size, *)
              usually -gradient
          iterate: torch.Tensor of shape (batch_size, *)
              usually the iterate of the considered algorithm

        Returns:
          update_direction: torch.Tensor, same shape as grad and iterate,
              s - iterate, where s is the vertex of the constraint most correlated
              with u
          max_step_size: torch.Tensor of shape (batch_size,)
              1. for a Frank-Wolfe step.
        """
        update_direction = -iterate.clone().detach()
        update_direction += self.alpha * torch.sign(grad)
        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)

    @torch.no_grad()
    def random_point(self, shape):
        """Returns a point of given shape uniformly at random from the constraint set."""
        z = torch.zeros(*shape)
        z.uniform_(-self.alpha, self.alpha)
        return z

    @torch.no_grad()
    def lmo_pairwise(self, grad, iterate, active_set):
        fw_direction = self.lmo(grad, iterate) + iterate.clone().detach()

        away_direction = min(self.active_set.keys(),
                             key=lambda v: torch.tensor(v).dot(grad))
        max_step = self.active_set[away_direction]
        away_direction = torch.tensor(away_direction)
        return fw_direction - away_direction, max_step

    def is_feasible(self, x, rtol=1e-5, atol=1e-7):
        return abs(x).reshape(x.size(0), -1).max(dim=-1)[0] <= self.alpha * (1. + rtol) + atol


class L1Ball(LpBall):
    p = 1

    @torch.no_grad()
    def lmo(self, grad, iterate):
        """Linear Maximization Oracle.
        Return s - iterate with s solving the linear problem

        ..math::
            max_{||s||_1 <= alpha} <grad, s>

        Args:
          grad: torch.Tensor of shape (batch_size, *)
              usually -gradient
          iterate: torch.Tensor of shape (batch_size, *)
              usually the iterate of the considered algorithm

        Returns:
          update_direction: torch.Tensor, same shape as grad and iterate,
              s - iterate, where s is the vertex of the constraint most correlated
              with u
          max_step_size: torch.Tensor of shape (batch_size,)
              1. for a Frank-Wolfe step.
        """
        update_direction = -iterate.clone().detach()
        abs_grad = abs(grad)
        batch_size = iterate.size(0)
        flatten_abs_grad = abs_grad.view(batch_size, -1)
        flatten_largest_mask = (flatten_abs_grad == flatten_abs_grad.max(-1, True)[0])
        largest = torch.where(flatten_largest_mask.view_as(abs_grad))

        update_direction[largest] += self.alpha * torch.sign(
            grad[largest])

        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)

    @torch.no_grad()
    def prox(self, x, step_size=None):
        """Projection onto the L1 ball.

        Args:
          x: torch.Tensor of shape (batchs_size, *)
            tensor to project
          step_size: Any
            Not used here
        
        Returns:
          p: torch.Tensor, same shape as x
            projection of x onto the L1 ball.
        """
        shape = x.shape
        flattened_x = x.reshape(shape[0], -1)
        # TODO vectorize this
        projected = [euclidean_proj_l1ball(row, s=self.alpha) for row in flattened_x]
        x = torch.stack(projected)
        return x.reshape(*shape)


class L2Ball(LpBall):
    p = 2

    @torch.no_grad()
    def prox(self, x, step_size=None):
        """Projection onto the L2 ball.

        Args:
          x: torch.Tensor of shape (batchs_size, *)
            tensor to project
          step_size: Any
            Not used here
        
        Returns:
          p: torch.Tensor, same shape as x
            projection of x onto the L2 ball.
        """
        norms = utils.bnorm(x)
        mask = norms > self.alpha
        projected = x.clone().detach()
        projected[mask] = self.alpha * utils.bdiv(projected[mask], norms[mask])
        return projected


    @torch.no_grad()
    def lmo(self, grad, iterate):
        """Linear Maximization Oracle.
        Return s - iterate with s solving the linear problem

        ..math::
            max_{||s||_2 <= alpha} <grad, s>

        Args:
          grad: torch.Tensor of shape (batch_size, *)
              usually -gradient
          iterate: torch.Tensor of shape (batch_size, *)
              usually the iterate of the considered algorithm

        Returns:
          update_direction: torch.Tensor, same shape as grad and iterate,
              s - iterate, where s is the vertex of the constraint most correlated
              with u
          max_step_size: torch.Tensor of shape (batch_size,)
              1. for a Frank-Wolfe step.
        """
        update_direction = -iterate.clone().detach()
        grad_norms = torch.norm(grad.view(grad.size(0), -1), p=2, dim=-1)
        update_direction += self.alpha * (grad.view(grad.size(0), -1).T
                                            / grad_norms).T.view_as(iterate)
        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)


def make_LpBall(alpha, p=1):
    if p == 1:
        return L1Ball(alpha)
    elif p == 2:
        return L2Ball(alpha)

    elif p == np.inf:
        return LinfBall(alpha)

    raise NotImplementedError("We have only implemented ord={1, 2, np.inf} for now.")


class Simplex:

    def __init__(self, alpha):
        if alpha >= 0:
            self.alpha = alpha
        else:
            raise ValueError("alpha must be a non negative number.")

    @torch.no_grad()
    def prox(self, x, step_size=None):
        shape = x.shape
        flattened_x = x.view(shape[0], -1)
        projected = [euclidean_proj_simplex(row, s=self.alpha) for row in flattened_x]
        x = torch.stack(projected)
        return x.view(*shape)

    @torch.no_grad()
    def lmo(self, grad, iterate):
        batch_size = grad.size(0)
        shape = iterate.shape
        max_vals, max_idx = grad.reshape(batch_size, -1).max(-1)

        update_direction = -iterate.clone().detach().reshape(batch_size, -1)
        update_direction[range(batch_size), max_idx] += self.alpha
        update_direction = update_direction.reshape(*shape)

        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)

    def is_feasible(self, x, rtol=1e-5, atol=1e-7):
        batch_size = x.size(0)
        reshaped_x = x.reshape(batch_size, -1)
        return torch.logical_and(reshaped_x.min(dim=-1)[0] + atol >= 0,
                                 reshaped_x.sum(-1) <= self.alpha * (1. + rtol) + atol) 


class NuclearNormBall:
    """
    Nuclear norm constraint, i.e. sum of absolute eigenvalues.
    Also known as the Schatten-1 norm.
    We consider the last two dimensions of the input are the ones we compute the Nuclear Norm on.
    """
    def __init__(self, alpha):
        if not 0. <= alpha:
            raise ValueError("Invalid constraint size alpha: {}".format(alpha))
        self.alpha = alpha

    @torch.no_grad()
    def lmo(self, grad, iterate):
        """
        Computes the LMO for the Nuclear Norm Ball on the last two dimensions.
        Returns :math: `s - $iterate$` where

          ..math::
            s = \argmax_u u^\top grad.

        Args:
          grad: torch.Tensor of shape (*, m, n)
          iterate: torch.Tensor of shape (*, m, n)
        Returns:
          update_direction: torch.Tensor of shape (*, m, n)
        """
        update_direction = -iterate.clone().detach()
        u, _, v = utils.power_iteration(grad)
        outer = u.unsqueeze(-1) * v.unsqueeze(-2)
        update_direction += self.alpha * outer
        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)

    @torch.no_grad()
    def prox(self, x, step_size=None):
        """
        Projection operator on the Nuclear Norm constraint set.
        """
        U, S, V = torch.svd(x)
        # Project S on the alpha-L1 ball
        ball = L1Ball(self.alpha)

        S_proj = ball.prox(S.view(-1, S.size(-1))).view_as(S)

        VT = V.transpose(-2, -1)
        return torch.matmul(U, torch.matmul(torch.diag_embed(S_proj), VT))

    def is_feasible(self, x, atol=1e-5, rtol=1e-5):
        norms = torch.linalg.norm(x, dim=(-2, -1), ord='nuc')
        return (norms <= self.alpha * (1. + rtol) + atol)


class GroupL1Ball:

    # TODO: init is shared with the penalty GroupL1 object. Factorize the code.
    def __init__(self, alpha, groups):
        if alpha >= 0:
            self.alpha = alpha
        else:
            raise ValueError("alpha must be nonnegative.")
        # TODO: implement ValueErrors
        # groups must be indices and non overlapping
        if not isinstance(groups[0], torch.Tensor):
            groups = [torch.tensor(group) for group in groups]
        while groups[0].dim() < 2:
            groups = [group.unsqueeze(-1) for group in groups]

        self.groups = []
        for g in groups:
            self.groups.append((...,) + tuple(g.T))

    def get_group_norms(self, x):
        """Compute the vector of L2 norms within groups"""
        group_norms = []
        for g in self.groups:
            subtensor = x[g]

            group_norms.append(torch.linalg.norm(subtensor, dim=-1))

        group_norms = torch.stack(group_norms, dim=-1)
        return group_norms

    @torch.no_grad()
    def lmo(self, grad, iterate):
        update_direction = -iterate.detach().clone()
        # find group with largest L2 norm
        group_norms = self.get_group_norms(grad)
        max_groups = torch.argmax(group_norms, dim=-1)

        for k, max_group in enumerate(max_groups):
            idx = (k, *self.groups[max_group])
            update_direction[idx] += (self.alpha * grad[idx]
                                      / group_norms[k, max_group])

        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)


    @torch.no_grad()
    def prox(self, x, step_size=None):
        """Proximal operator for the GroupL1 constraint"""

        group_norms = self.get_group_norms(x)
        l1ball = L1Ball(self.alpha)
        normalized_group_norms = l1ball.prox(group_norms)

        output = x.detach().clone()

        # renormalize each group
        for k, g in enumerate(self.groups):
            renorm = normalized_group_norms[:, k] / group_norms[:, k]
            renorm[torch.isnan(renorm)] = 1.
            output[g] = utils.bmul(output[g], renorm)

        return output

    def is_feasible(self, x, rtol=1e-5, atol=1e-7):
        group_norms = self.get_group_norms(x)
        return torch.linalg.norm(group_norms, ord=1, dim=-1) <= (self.alpha * (1. + rtol)
                                                                 + atol)


class Box:
    """
    Box constraint.
    Args:
        a: float or None
        min of the box constraint
        b: float or None
        max of the box constraint
    """
    def __init__(self, a=None, b=None):
        """
          """

        if a is None and b is None:
            raise ValueError("One of a, b should not be None.")
        if a is None:
            a = -np.inf
        elif b is None:
            b = np.inf
        else:
            if b < a:
                raise ValueError(f"This constraint supposes that a <= b. Got {a}, {b}.")
        self.a = a
        self.b = b

    def prox(self, x, step_size=None):
        """Projection operator on the constraint.
        Args:
          x: torch.Tensor
          step_size: Any
        Returns:
          x_thresh: torch.Tensor
            x clamped between a and b.
        """
        return torch.clamp(x, min=self.a, max=self.b)

    def is_feasible(self, x, rtol=1e-5, atol=1e-7):
        reshaped_x = x.reshape(x.size(0), -1)
        return torch.logical_and(reshaped_x.min(-1)[0] >= self.a * (1. + rtol) - atol,
                                 reshaped_x.max(-1)[0] <= self.b * (1. + rtol) + atol)


class Cone:
    """
    Represents second order cones of revolution centered in vector `u` (batch-wise), and angle :math: `\hat alpha`.
    This constraint therefore really represents a batch of cones, which share the same half-angle.
    The are all pointed in 0 (the origin).
    Formally, the set is the following:
    ..math::
        \{x \in R^d,~ \|(uu^\top - Id)x\| \leq \alpha u^\top x \}
    Note that :math: `\cos(\hat \alpha) = 1 / (1 + \alpha^2)`.
    The standard second order cone (ice-cream cone) is given by
    `u = (0, ..., 0, 1)`, `cos_alpha=.5`.
    Args:
        u: torch.Tensor
        batch-wise directions centering the cones
        cos_angle: float
        cosine of the half-angle of the cone.
    """
    def __init__(self, u, cos_angle=.05):
        batch_size = u.size(0)
        # normalize the cone directions
        self.directions = utils.bmul(u, 1. / torch.norm(u.reshape(batch_size, -1), dim=-1))
        self.cos_angle = cos_angle
        self.alpha = np.sqrt(1. / cos_angle - 1)

    def proj_u(self, x, step_size=None):
        """
        Projects x on self.directions batch-wise
        Args:
          x: torch.Tensor of shape (batch_size, *)
            vectors to project
          step_size: Any
            Not used
        Returns:
          proj_x: torch.Tensor of shape (batch_size, *)
            batch-wise projection of x onto self.directions
        """
        
        return utils.bmul(utils.bdot(x, self.directions), self.directions)


    @torch.no_grad()
    def prox(self, x, step_size=None):
        """
        Projects `x` batch-wise onto the cone constraint.
        Args:
          x: torch.Tensor of shape (batch_size, *)
            batch of vectors to project
          step_size: Any
            Not used
        Returns:
          proj_x: torch.Tensor of shape (batch_size, *)
            batch-wise projection of `x` onto the cone constraint.
        """
        batch_size = x.size(0)
        uTx = utils.bdot(self.directions, x)
        p_u = self.proj_u(x)
        p_orth_u = x - p_u
        norm_p_orth_u = torch.norm(p_orth_u.reshape(batch_size, -1), dim=-1)
        identity_idx = (norm_p_orth_u <= self.alpha * uTx)
        zero_idx = (self.alpha * norm_p_orth_u <= - uTx)
        project_idx = ~torch.logical_or(identity_idx, zero_idx)

        res = x.detach().clone()
        res[zero_idx] = 0.
        res[project_idx] = utils.bmul((self.alpha * norm_p_orth_u[project_idx] + uTx[project_idx]) / (1. + self.alpha ** 2), 
                                      (self.alpha * utils.bmul(p_orth_u[project_idx], 1 / norm_p_orth_u[project_idx])
                                       + self.directions[project_idx]))
        return res

    def is_feasible(self, x, rtol=1e-5, atol=1e-7):
        cosines = utils.bdot(x, self.directions)
        return abs(cosines) >= utils.bnorm(x) * self.cos_angle * (1. + rtol) + atol
