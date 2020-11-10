from copy import deepcopy
from collections import defaultdict

import torch

import numpy as np
from scipy.stats import expon
from torch.distributions import Laplace, Normal
# TODO: Add projections to the constraints, and write ProjectedOptimizer wrapper/decorator

"""This uses an API similar to the one for
the COPT project, https://github.com/openopt/copt."""


def euclidean_proj_simplex(v, s=1.):
    r""" Compute the Euclidean projection on a positive simplex
  Solves the optimization problem (using the algorithm from [1]):
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
    rho = torch.nonzero(u * torch.arange(1, n + 1, device=v.device) > (cssv - s))[-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = torch.clamp(v - theta, min=0)
    return w


def euclidean_proj_l1ball(v, s=1.):
    """ Compute the Euclidean projection on a L1-ball
  Solves the optimization problem (using the algorithm from [1]):
      min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
  Parameters
  ----------
  v: (n,) numpy array,
      n-dimensional vector to project
  s: float, optional, default: 1,
      radius of the L1-ball
  Returns
  -------
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

    def fw_gap(self, grad, iterate):
        update_direction, _ = self.lmo(-grad, iterate)
        return (-grad * update_direction).sum()

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


class LinfBall(LpBall):
    p = np.inf

    def prox(self, x, step_size=None, batch=False):
        if torch.max(abs(x)) <= self.alpha:
            return x
        return torch.clamp(x, min=-self.alpha, max=self.alpha)

    def lmo(self, grad, iterate, batch=False):
        update_direction = -iterate.clone().detach()
        update_direction += self.alpha * torch.sign(grad)
        return update_direction, 1.

    def random_point(self, shape):
        """Returns a point of given shape uniformly at random from the constraint set."""
        return self.alpha * torch.FloatTensor(*shape).uniform_(-1, 1)

    def lmo_pairwise(self, grad, iterate, active_set, batch=False):
        fw_direction = self.lmo(grad, iterate, batch) + iterate.clone().detach()

        away_direction = min(self.active_set.keys(),
                             key=lambda v: torch.tensor(v).dot(grad))
        max_step = self.active_set[away_direction]
        away_direction = torch.tensor(away_direction)
        return fw_direction - away_direction, max_step
        

class L1Ball(LpBall):
    p = 1

    def lmo(self, grad, iterate, batch=False):
        """Returns s-x, s solving the linear problem
        max_{\|s\|_1 <= \\alpha} \\langle grad, s\\rangle
        """
        update_direction = -iterate.clone().detach()
        abs_grad = abs(grad)
        if batch:
            batch_size = iterate.size(0)
            flatten_abs_grad = abs_grad.view(batch_size, -1)
            flatten_largest_mask = (flatten_abs_grad == flatten_abs_grad.max(-1, True)[0])
            largest = torch.where(flatten_largest_mask.view_as(abs_grad))
        else:
            largest = torch.where(abs_grad == abs_grad.max())

        update_direction[largest] += self.alpha * torch.sign(
            grad[largest])

        return update_direction, 1.

    def prox(self, x, step_size=None, batch=False):
        shape = x.shape
        if batch:
            flattened_x = x.view(shape[0], -1)
            projected = [self.prox(row.view(-1), step_size, batch=False)
                         for row in flattened_x]
            x = torch.stack(projected)
        else:
            x = euclidean_proj_l1ball(x.view(-1), self.alpha)
        return x.view(*shape)


# TODO: #1 Fix L2 Ball
class L2Ball(LpBall):
    p = 2

    def prox(self, x, step_size=None, batch=False):
        if batch:
            shape = x.shape
            norms = torch.norm(x.view(shape[0], -1), p=2, dim=-1)
            mask = norms > self.alpha
            projected = x.clone().detach()
            projected[mask] = (projected[mask].T / norms[mask]).T
            return self.alpha * projected.view(shape)

        norm = torch.sqrt((x ** 2).sum())
        if norm <= self.alpha:
            return x
        return self.alpha * x / norm

    def lmo(self, grad, iterate, batch=False):
        """
        Returns s-x, s solving the linear problem
        max_{\|s\|_2 <= \\alpha} \\langle grad, s\\rangle
        """
        update_direction = iterate.clone().detach()
        if batch:
            grad_norms = torch.norm(grad.view(grad.size(0), -1), p=2, dim=-1)
            update_direction += self.alpha * (grad.view(grad.size(0), -1).T
                                              / grad_norms).T.view_as(iterate)
        else:
            grad_norm = torch.sqrt((grad ** 2).sum())
            update_direction += self.alpha * grad / grad_norm
        return update_direction, 1.


def make_LpBall(alpha, p=1):
    if p == 1:
        return L1Ball(alpha)
    elif p == 2:
        return L2Ball(alpha)

    elif p == np.inf:
        return LinfBall(alpha)
    
    raise NotImplementedError("We have only implemented ord={1, 2, np.inf} for now.")
    

class Simplex:

    def prox(self, x, step_size=None):
        shape = x.shape
        x = euclidean_proj_simplex(x.view(-1), self.alpha)
        return x.view(*shape)

    def lmo(self, grad, iterate):
        largest_coordinate = torch.where(grad == grad.max())

        update_direction = -iterate.clone().detach()
        update_direction[largest_coordinate] += self.alpha * torch.sign(
            grad[largest_coordinate]
        )

        return update_direction, 1.
