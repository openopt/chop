import pytest
import numpy as np
import torch

import chop


@pytest.mark.parametrize('alpha', [0., 1e-3, 1.])
def test_groupL1_unit_groups_against_lasso_1d(alpha):
    n_features = 4
    groups = torch.arange(n_features)
    lasso = chop.penalties.L1(alpha)
    groupLasso = chop.penalties.GroupL1(alpha, groups)

    batch_size = 3
    data = torch.rand(batch_size, n_features)
    assert torch.allclose(lasso(data), groupLasso(data)), '__call__'

    assert torch.allclose(lasso.prox(data, 1.), groupLasso.prox(data, 1.)), 'prox'

    if alpha == 0.:
        assert torch.allclose(data, groupLasso.prox(data, 1.))


def test_zero_data_L1_groupL1_prox():
    batch_size, n_features = 3, 4
    data = torch.zeros(batch_size, n_features)
    alpha = 1
    groups = torch.arange(n_features)
    lasso = chop.penalties.L1(alpha)
    groupLasso = chop.penalties.GroupL1(alpha, groups)
    
    assert (lasso.prox(data, 1.) == data).all()
    assert (groupLasso.prox(data, 1.) == data).all()


def test_groupL1_1d():
    groups = [(0, 1),
              (2, 3)]
    alpha = 1.
    penalty = chop.penalties.GroupL1(alpha, groups)

    batch_size = 3
    data = torch.rand(batch_size, 4)

    correct_result = alpha * (torch.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
                              + torch.sqrt(data[:, 2] ** 2 + data[:, 3] ** 2))
    assert torch.allclose(penalty(data), correct_result)


def test_groupL1_2d():
    groups = [((0, 0), (0, 1), (1, 0), (1, 1)),
              ((2, 0), (2, 1), (3, 0), (3, 1))]
    alpha = 1.
    penalty = chop.penalties.GroupL1(alpha, groups)
    batch_size = 3

    data = torch.rand(batch_size, 4, 2)

    correct_result = alpha * (torch.sqrt(data[:, 0, 0] ** 2 + data[:, 1, 0] ** 2
                                         + data[:, 0, 1] ** 2 + data[:, 1, 1] ** 2)
                              + torch.sqrt(data[:, 2, 0] ** 2 + data[:, 3, 0] ** 2
                                           + data[:, 2, 1] ** 2 + data[:, 3, 1] ** 2))
    assert torch.allclose(penalty(data), correct_result), '__call__'



@pytest.mark.parametrize("penalty", [chop.penalties.GroupL1(1., np.array_split(np.arange(16), 5)),
                                     chop.penalties.L1(1.)])
def test_three_inequality(penalty):
    """Test the L1 prox using the three point inequality
    The three-point inequality is described e.g., in Lemma 1.4
    in "Gradient-Based Algorithms with Applications to Signal
    Recovery Problems", Amir Beck and Marc Teboulle
    """
    n_features = 16
    batch_size = 3

    for _ in range(10):
        z = torch.rand(batch_size, n_features)
        u = torch.rand(batch_size, n_features)
        xi = penalty.prox(z, 1.)

        lhs = 2 * (penalty(xi) - penalty(u))
        rhs = (
            torch.linalg.norm(u - z, dim=-1) ** 2
            - torch.linalg.norm(u - xi, dim=-1) ** 2
            - torch.linalg.norm(xi - z, dim=-1) ** 2
        )
        assert (lhs <= rhs).all(), penalty
