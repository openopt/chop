import torch
from torch.optim import Optimizer

import numpy as np

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
    rho = torch.nonzero(u * np.arange(1, n + 1) > (cssv - s))[-1]
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
    w *= np.sign(v)
    return w


class LpBall:
    def __init__(self, alpha):
        if not 0. <= alpha:
            raise ValueError("Invalid constraint size alpha: {}".format(alpha))
        self.alpha = alpha

    def fw_gap(self, grad, iterate):
        update_direction, _ = self.lmo(-grad, iterate)
        return (-grad * update_direction).sum()


class LinfBall(LpBall):
    def __init__(self, alpha):
        super(LinfBall, self).__init__(alpha)
        self.p = np.inf

    def prox(self, x, step_size=None):
        if torch.max(abs(x)) <= self.alpha:
            return x
        return torch.clamp(x, min=-self.alpha, max=self.alpha)

    def lmo(self, grad, iterate):
        update_direction = -iterate.clone().detach()
        update_direction += self.alpha * torch.sign(grad)
        return update_direction, 1.


class L1Ball(LpBall):

    def __init__(self, alpha):
        super(L1Ball, self).__init__(alpha)
        self.p = 1

    def lmo(self, grad, iterate):
        """Returns s-x, x solving the linear problem
        max_{\|s\|_1 <= \\alpha} \\langle grad, s\\rangle
        """
        update_direction = -iterate.clone().detach()
        abs_grad = abs(grad)
        largest_coordinate = torch.where(abs_grad == abs_grad.max())

        update_direction[largest_coordinate] += self.alpha * torch.sign(
            grad[largest_coordinate])

        return update_direction, 1.

    def prox(self, x, step_size=None):
        return euclidean_proj_l1ball(x, self.alpha)

class L2Ball(LpBall):
    def __init__(self, alpha):
        super(L2Ball, self).__init__(alpha)
        self.p = 2

    def prox(self, x, step_size=None):
        norm = torch.norm(x)
        if norm <= self.alpha:
            return x
        return x / torch.norm(x)

    def lmo(self, grad, iterate):
        update_direction = iterate.clone().detach()
        grad_norm = (grad ** 2).sum()
        update_direction += self.alpha * grad / grad_norm
        return update_direction, 1


def make_LpBall(alpha, p=1):
    if p == 1:
        return L1Ball(alpha)
    elif p == 2:
        return L2Ball(alpha)

    elif p == np.inf:
        return LinfBall(alpha)
    
    raise NotImplementedError("We have only implemented ord={1, 2, np.inf} for now.")
    