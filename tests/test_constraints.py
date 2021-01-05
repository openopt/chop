from chop.image import group_patches
import torch
import chop.constraints as constraints
import pytest


def test_nuclear_norm():

    batch_size = 8
    channels = 3
    m = 32
    n = 35
    alpha = 1.
    constraint = constraints.NuclearNormBall(alpha)

    grad = torch.rand(batch_size, channels, m, n)
    iterate = torch.rand(batch_size, channels, m, n)
    constraint.lmo(-grad, iterate)
    constraint.prox(iterate - .1 * grad)


@pytest.mark.parametrize('constraint', [constraints.L1Ball,
                                        constraints.L2Ball,
                                        constraints.LinfBall,
                                        constraints.Simplex,
                                        constraints.NuclearNormBall])
def test_projections(constraint):
    """Tests that projections are true projections:
    ..math::
        p\circ p = p
    """
    batch_size = 8
    alpha = 1.
    prox = constraint(alpha).prox

    for _ in range(10):
        data = torch.rand(batch_size, 3, 32, 32)

        proj_data = prox(data)
        # SVD reconstruction doesn't do better than 1e-5
        assert prox(proj_data).allclose(proj_data, atol=1e-5)


def test_GroupL1LMO():
    batch_size = 8
    alpha = 1.
    groups = group_patches()
    constraint = constraints.GroupL1Ball(alpha, groups)
    data = torch.rand(batch_size, 3, 32, 32)
    grad = torch.rand(batch_size, 3, 32, 32)
    
    constraint.lmo(-grad, data)
    
