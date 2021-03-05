"""
Penalties.
=========
This module contains classes representing penalties / regularizers.
They function batch-wise, similar to objects in `chop.constraints`.
Reshaping will be of order if the penalties are used on the parameters of a model.

Code inspired from https://github.com/openopt/copt/.

The proximal operators are derived e.g. in https://www.di.ens.fr/~fbach/opt_book.pdf.
"""

from numbers import Number
from numpy.core.fromnumeric import nonzero
import torch
import torch.nn.functional as F

from chop import utils


class L1:
    """L1 penalty. Batch-wise function. For each element in the batch,
    the L1 penalty is given by
    ..math::
        \Omega(x) = \alpha \|x\|_1
    """

    def __init__(self, alpha: float):
        """
        Args:
          alpha: float
            Size of the penalty. Must be non-negative.
        """
        if alpha < 0:
            raise ValueError("alpha must be non negative.")
        self.alpha = alpha

    def __call__(self, x):
        """
        Returns the value of the penalty on x, batch_size.

        Args:
          x: torch.Tensor
            x has shape (batch_size, *)
        """
        batch_size = x.size(0)
        return self.alpha * abs(x.view(batch_size, -1)).sum(dim=-1)

    def prox(self, x, step_size=None):
        """Proximal operator for the L1 norm penalty. This is given by soft-thresholding.

        Args:
          x: torch.Tensor
            x has shape (batch_size, *)
          step_size: float or torch.Tensor of shape (batch_size,)
        
        """
        if isinstance(step_size, Number):
            step_size = step_size * torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        return utils.bmul(torch.sign(x), F.relu(abs(x) - self.alpha * step_size.view((-1,) + (1,) * (x.dim() - 1))))


class GroupL1:
    """
    Group LASSO penalty. Batch-wise function.
    """

    def __init__(self, alpha, groups):
        """
        Args:
          alpha: float
            Size of the penalty. Must be non-negative.

          groups: iterable of iterables
            Each element of groups will be used to index the given tensor to compute
            the penalty on. See example.

        Examples:
          Our input is of shape (batch_size, 4), and we want to split the features in two groups.
          The first contains the first two features, and the second the latter two. This is done by:
            $ groups = [(0, 1), (2, 3)]

          In this case, since the groups are of equal size, we could have used
            $ groups = torch.tensor([[0, 1],
            $                        [2, 3]])

          If the input is of shape (batch_size, 4, 2), and we want to split
          our features in 2 groups (left half and right half of the image), then each group
          is an iterable over the coordinates contained in it.

            $ groups = [((0, 0), (0, 1), (1, 0), (1, 1)),
            $           ((2, 0), (2, 1), (3, 0), (3, 1))]

          The same convention is used for higher dimension inputs.
          If the provided coordinates are of smaller dimension, they will be prepended
          by an Ellipsis.

        Todo:
          * Test the Ellipsis behavior.
        """
        self.alpha = alpha

        # TODO: implement ValueErrors
        # groups must be indices and non overlapping
        if not isinstance(groups[0], torch.Tensor):
            groups = [torch.tensor(group) for group in groups]
        while groups[0].dim() < 2:
            groups = [group.unsqueeze(-1) for group in groups]
            
        self.groups = groups

    def __call__(self, x):
        group_norms = torch.stack([torch.linalg.norm(x[(...,) + tuple(g.T)],
                                                     dim=-1)
                                   for g in self.groups])

        return self.alpha * group_norms.sum(dim=0)

    @torch.no_grad()
    def prox(self, x, step_size=None):
        """
        Returns the proximal operator for the (non overlapping) Group L1 norm.
        Args:
          x: torch.Tensor of shape (batch_size, *)

          step_size: float or torch.Tensor of shape (batch_size,)

        """
        out = x.detach().clone()
        if isinstance(step_size, Number):
            step_size *= torch.ones(x.size(0), dtype=x.dtype, device=x.device)

        for g in self.groups:
            norm = torch.linalg.norm(x[(...,) + tuple(g.T)].view(x.size(0), -1), dim=-1)
            nonzero_norm = torch.nonzero(norm)
            out[(nonzero_norm, ...) + tuple(g.T)] = utils.bmul(out[(nonzero_norm, ...) + tuple(g.T)],
                                                               F.relu(1 - utils.bmul(self.alpha * step_size[nonzero_norm],
                                                                                     1. / norm[nonzero_norm])))
        return out
