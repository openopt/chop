from chop.utils.image import group_patches
import torch
import chop.constraints as constraints
import pytest
from chop import utils

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
                                        constraints.NuclearNormBall,
                                        constraints.GroupL1Ball,
                                        constraints.Cone])
def test_projections(constraint):
    """Tests that projections are true projections:
    ..math::
        p\circ p = p
    """
    batch_size = 8
    alpha = 1.
    if constraint == constraints.GroupL1Ball:
        groups = group_patches()
        prox = constraint(alpha, groups).prox
    elif constraint == constraints.Cone:
        directions = torch.rand(batch_size, 3, 32, 32)
        prox = constraint(directions, cos_angle=.2).prox
    else:
        prox = constraint(alpha).prox

    for _ in range(10):
        data = torch.rand(batch_size, 3, 32, 32)

        proj_data = prox(data)
        # SVD reconstruction doesn't do better than 1e-5
        double_proj = prox(proj_data)
        assert double_proj.allclose(proj_data, atol=1e-5), (double_proj, proj_data)


def test_GroupL1LMO():
    batch_size = 2
    alpha = 1.
    groups = group_patches(x_patch_size=2, y_patch_size=2, x_image_size=6, y_image_size=6)
    constraint = constraints.GroupL1Ball(alpha, groups)
    data = torch.rand(batch_size, 3, 6, 6)
    grad = torch.rand(batch_size, 3, 6, 6)

    constraint.lmo(-grad, data)


def test_groupL1Prox():
    batch_size = 2
    alpha = 10
    groups = group_patches(x_patch_size=2, y_patch_size=2, x_image_size=6, y_image_size=6)
    constraint = constraints.GroupL1Ball(alpha, groups)
    data = torch.rand(batch_size, 3, 6, 6)

    constraint.prox(-data, step_size=.3)


def test_cone_constraint():
    # Standard second order cone
    u = torch.tensor([[0., 0., 1.]])
    cos_alpha = .5

    cone = constraints.Cone(u, cos_alpha)

    for inp, correct_prox in [(torch.tensor([[1., 0, 0]]), torch.tensor([[.5, 0, .5]])),
                              (torch.tensor([[0, 1., 0]]), torch.tensor([[0, .5, .5]])),
                              (u, u),
                              (-u, torch.zeros_like(u))
                              ]:
        assert cone.prox(inp).eq(correct_prox).all()


    # Moreau decomposition: x = proj_x + (x - proj_x) where these two vectors are orthogonal
    for _ in range(10):
        x = torch.rand(*u.shape)
        proj_x = cone.prox(x)
        assert utils.bdot(x - proj_x, proj_x).allclose(torch.zeros_like(x), atol=4e-7)
