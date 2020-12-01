"""
This module contains various penalties / regularizers.
They function batch-wise, similar to objects in `chop.constraints`.

Code inspired from https://github.com/openopt/copt/
"""

import torch
import torch.functional as F

from chop import utils


class L1:
    """L1 penalty. Batch-wise function."""

    def __init__(self, alpha):
        if alpha < 0:
            raise ValueError("alpha must be non negative.")
        self.alpha = alpha

    def __call__(self, x):
        batch_size = x.size(0)
        return abs(x.view(batch_size, -1)).sum(dim=-1)

    def prox(self, x, step_size=None):
        """Prox operator for L1 norm. This is given by soft-thresholding."""
        return torch.sign(x) * F.relu(abs(x) - step_size)


class GroupL1:
    """
    Group LASSO penalty. Batch-wise function.
    """

    def __init__(self, alpha, groups):
        self.alpha = alpha

        # TODO: implement ValueErrors
        # groups must be indices
        self.groups = groups

    def __call__(self, x):
        if x.dim() == 2:
            group_norms = torch.stack([torch.linalg.norm(x[:, g], dim=-1)
                                       for g in self.groups])
        else:
            group_norms = torch.stack([torch.linalg.norm(x[(Ellipsis,) + tuple(zip(*g))],
                                                         dim=-1)
                                       for g in self.groups])

        return self.alpha * group_norms.sum(dim=0)

    @torch.no_grad()
    def prox(self, x, step_size=None):
        out = x.detach().clone()

        if x.dim() == 2:
            for g in self.groups:
                norm = torch.linalg.norm(x[:, g], dim=-1)

                mask = norm > self.alpha * step_size

                out[mask, g] -= step_size * self.alpha * utils.bmul(out[mask, g], 1. / norm)
                out[~mask, g] = 0

            return out

        for g in self.groups:
            idx = tuple(zip(*g))
            norm = torch.linalg.norm(x[(slice(None),) + idx], dim=-1)
            mask = norm > self.alpha * step_size

            out[mask][slice(None, ) + idx] -= step_size * self.alpha * utils.bmul(out[mask][slice(None, ) + idx], 1. / norm)
            out[~mask][slice(None, ) + idx] = 0

        return out
        