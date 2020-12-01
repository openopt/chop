import torch

import chop


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
    assert torch.allclose(penalty(data), correct_result)
