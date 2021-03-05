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

import torch

import numpy as np
from scipy.stats import expon
from torch.distributions import Laplace, Normal
from chop import utils


@torch.no_grad()
def get_avg_init_norm(layer, param_type=None, p=2, repetitions=100):
    """Computes the average norm of default layer initialization"""
    output = 0
    for _ in range(repetitions):
        layer.reset_parameters()
        output += torch.norm(getattr(layer, param_type), p=p).item()
    return float(output) / repetitions


@torch.no_grad()
def create_lp_constraints(model, p=2, value=300, mode='initialization'):
    """Create LpBall constraints for each layer of model, and value depends on mode (either radius or
    factor to multiply average initialization norm with)"""
    constraints = []

    # Compute average init norms if necessary
    init_norms = dict()
    if mode == 'initialization':
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                for param_type in [entry for entry in ['weight', 'bias'] if (hasattr(layer, entry) and
                                                                             type(getattr(layer, entry)) != type(
                            None))]:
                    param = getattr(layer, param_type)
                    shape = param.shape

                    avg_norm = get_avg_init_norm(layer, param_type=param_type, p=2)
                    if avg_norm == 0.0:
                        # Catch unlikely case that weight/bias is 0-initialized (e.g. BatchNorm does this)
                        avg_norm = 1.0
                    init_norms[shape] = avg_norm

    for name, param in model.named_parameters():
        if mode == 'radius':
            constraint = make_LpBall(value, p=p)
        elif mode == 'initialization':
            alpha = value * init_norms[param.shape]
            constraint = make_LpBall(alpha, p=p)
        else:
            raise ValueError(f"Unknown mode {mode}")
        constraints.append(constraint)
    return constraints


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


class LinfBall(LpBall):
    p = np.inf

    @torch.no_grad()
    def prox(self, x, step_size=None):
        if torch.max(abs(x)) <= self.alpha:
            return x
        return torch.clamp(x, min=-self.alpha, max=self.alpha)

    @torch.no_grad()
    def lmo(self, grad, iterate):
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
        

class L1Ball(LpBall):
    p = 1

    @torch.no_grad()
    def lmo(self, grad, iterate):
        """Returns s-x, s solving the linear problem
        max_{\|s\|_1 <= \\alpha} \\langle grad, s\\rangle
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
        shape = x.shape
        flattened_x = x.view(shape[0], -1)
        # TODO vectorize this
        projected = [euclidean_proj_l1ball(row) for row in flattened_x]
        x = torch.stack(projected)
        return x.view(*shape)


class L2Ball(LpBall):
    p = 2

    @torch.no_grad()
    def prox(self, x, step_size=None):
        shape = x.shape
        norms = utils.bnorm(x)
        mask = norms > self.alpha
        projected = x.clone().detach()
        projected[mask] = utils.bdiv(projected[mask], norms[mask])
        return self.alpha * projected.view(shape)


    @torch.no_grad()
    def lmo(self, grad, iterate):
        """
        Returns s-x, s solving the linear problem
        max_{\|s\|_2 <= \\alpha} \\langle grad, s\\rangle
        """
        update_direction = iterate.clone().detach()
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
        x = euclidean_proj_simplex(x.view(-1), self.alpha)
        return x.view(*shape)

    @torch.no_grad()
    def lmo(self, grad, iterate):
        largest_coordinate = torch.where(grad == grad.max())

        update_direction = -iterate.clone().detach()
        update_direction[largest_coordinate] += self.alpha * torch.sign(
            grad[largest_coordinate]
        )

        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)


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
            s = \argmin_u u^\top grad.

        Args:
          grad: torch.Tensor of shape (*, m, n)

          iterate: torch.Tensor of shape (*, m, n)

        Returns:
          update_direction: torch.Tensor of shape (*, m, n)
        """
        update_direction = -iterate.clone().detach()
        u, s, v = utils.power_iteration(grad)
        outer = u.unsqueeze(-1) * v.unsqueeze(-2)
        update_direction += self.alpha * utils.bmul(s, outer)
        return update_direction, torch.ones(iterate.size(0), device=iterate.device, dtype=iterate.dtype)

    @torch.no_grad()
    def prox(self, x, step_size=None):
        """
        Projection operator on the Nuclear Norm constraint set.
        """

        U, S, V = torch.svd(x)
        # Project S on the alpha-simplex
        simplex = Simplex(self.alpha)

        S_proj = simplex.prox(S.view(-1, S.size(-1))).view_as(S)
        
        VT = V.transpose(-2, -1)
        return torch.matmul(U, torch.matmul(torch.diag_embed(S_proj), VT))


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


class Box:
    """
    Box constraint.


    Args:
        a: float
        min of the box constraint
        b: float
        max of the box constraint

    """
    def __init__(self, a, b):
        """
          """
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
        return torch.clamp(x, self.a, self.b)


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
