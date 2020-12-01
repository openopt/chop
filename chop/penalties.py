"""
This module contains various penalties / regularizers.
They function batch-wise, similar to objects in `chop.constraints`.

Code inspired from https://github.com/openopt/copt/
"""

import torch


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
            group_norms = torch.stack([torch.linalg.norm(x[(slice(None),) + tuple(zip(*g))],
                                                         dim=-1)
                                       for g in self.groups])

        return self.alpha * group_norms.sum(dim=0)

