"""Testing our adversarial attacks"""
import pytest
import torch
from torch import nn


import numpy as np

import chop
from chop import optim
from chop.adversary import Adversary


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(25 * 25, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1)
        return self.linear(x).view(batch_size, -1)


@pytest.mark.parametrize('algorithm', [optim.minimize_pgd, optim.minimize_pgd_madry,
                                       optim.minimize_frank_wolfe])
@pytest.mark.parametrize('step_size', [1, .5, .1, .05, .001, 0.])
@pytest.mark.parametrize('p', [1, 2, np.inf])
def test_adversary_synthetic_data(algorithm, step_size, p):
    # Setup
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data = torch.rand((1, 25, 25))
    target = torch.zeros(1).long()

    data = data.to(device)
    target = target.to(device)

    model = LinearModel()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    constraint = chop.constraints.make_LpBall(alpha=1., p=p)

    adv = Adversary(algorithm)

    # Get nominal loss
    output = model(data)
    loss = criterion(output, target)

    # Algorithm arguments:
    if algorithm == optim.minimize_pgd:
        alg_kwargs = {
            'prox': constraint.prox,
            'max_iter': 50
        }
    elif algorithm == optim.minimize_pgd_madry:
        alg_kwargs = {
            'prox': constraint.prox,
            'lmo': constraint.lmo,
            'max_iter': 50,
            'step': 2. * constraint.alpha / 50
        }

    elif algorithm == optim.minimize_frank_wolfe:
        alg_kwargs = {
            'lmo': constraint.lmo,
            'step': 'sublinear',
            'max_iter': 50
        }

    # Run perturbation
    adv_loss, delta = adv.perturb(data, target, model, criterion, **alg_kwargs)


@pytest.mark.parametrize('algorithm', [optim.minimize_pgd, optim.minimize_pgd_madry,
                                       optim.minimize_frank_wolfe
                                       ])
@pytest.mark.parametrize('step', [1., .5, .1, .05, None])
@pytest.mark.parametrize('p', [1, 2, np.inf])
@pytest.mark.parametrize('model_filename', ["mnist_lenet5_clntrained.pt", "mnist_lenet5_advtrained.pt"])
def test_adversary_mnist(algorithm, step, p, model_filename):

    import os
    from advertorch.test_utils import LeNet5
    from advertorch_examples.utils import get_mnist_test_loader
    from advertorch_examples.utils import TRAINED_MODEL_PATH

    # Setup
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    model = LeNet5()
    model_path = os.path.join(TRAINED_MODEL_PATH, model_filename)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    constraint = chop.constraints.make_LpBall(alpha=1., p=p)

    test_loader = get_mnist_test_loader(batch_size=1, shuffle=True)
    # Just get first batch
    for data, target in test_loader:
        break

    data, target = data.to(device), target.to(device)

    adv = Adversary(algorithm)

    if algorithm == optim.minimize_pgd:
        alg_kwargs = {
            'prox': constraint.prox,
            'max_iter': 50,
            'step': step
        }
    elif algorithm == optim.minimize_pgd_madry:
        alg_kwargs = {
            'prox': constraint.prox,
            'lmo': constraint.lmo,
            'max_iter': 50,
            'step': step
        }

    elif algorithm == optim.minimize_frank_wolfe:
        alg_kwargs = {
            'lmo': constraint.lmo,
            'step': step if step else 'sublinear',
            'max_iter': 50
        }

    # Run and log perturbation
    adv_loss, delta = adv.perturb(data, target, model, criterion,
                                  **alg_kwargs)
