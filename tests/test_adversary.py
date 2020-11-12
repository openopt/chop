"""Testing our adversarial attacks"""
import pytest
import shutil
import torch
from torch import nn


import numpy as np
from cox.store import Store

import constopt
from constopt import stochastic
from constopt.adversary import Adversary


OUT_DIR = "logging/tests/test_adversary/"
shutil.rmtree(OUT_DIR, ignore_errors=True)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(25 * 25, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1)
        return self.linear(x).view(batch_size, -1)


@pytest.mark.parametrize('algorithm', [stochastic.PGD, stochastic.PGDMadry,
                                       stochastic.FrankWolfe, stochastic.MomentumFrankWolfe])
@pytest.mark.parametrize('step_size', [1, .5, .1, .05, .001, 0.])
@pytest.mark.parametrize('p', [1, 2, np.inf])
def test_adversary_synthetic_data(algorithm, step_size, p):
    # Setup
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    data = torch.rand((1, 25, 25))
    target = torch.zeros(1).long()

    data = data.to(device)
    target = target.to(device)

    model = LinearModel()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    constraint = constopt.constraints.make_LpBall(alpha=1., p=p)

    adv = Adversary(data.shape, constraint, algorithm, device=device)
    optimizer = adv.optimizer
    # Logging

    store = Store(OUT_DIR)
    store.add_table('metadata', {'algorithm': str, 'step-size': float, 'p': float})

    store['metadata'].append_row({'algorithm': optimizer.name, 'step-size': step_size,
                                  'p': p})
    table_name = "L" + str(int(p)) + " ball" if p != np.inf else "Linf Ball"
    store.add_table(table_name, {'func_val': float, 'FW gap': float,
                                                  'norm delta': float})

    # Get nominal loss
    output = model(data)
    loss = criterion(output, target)
    # Run perturbation
    adv_loss, delta = adv.perturb(data, target, model, criterion, step_size, iterations=100,
                                  tol=1e-7, store=store)


@pytest.mark.parametrize('algorithm', [stochastic.PGD, stochastic.PGDMadry,
                                       stochastic.FrankWolfe, stochastic.MomentumFrankWolfe])
@pytest.mark.parametrize('step_size', [1, .5, .1, .05, .001, 0.])
@pytest.mark.parametrize('p', [1, 2, np.inf])
@pytest.mark.parametrize('random_init', [False, True])
@pytest.mark.parametrize('model_filename', ["mnist_lenet5_clntrained.pt", "mnist_lenet5_advtrained.pt"])
def test_adversary_mnist(algorithm, step_size, p, random_init, model_filename):

    import os
    from advertorch.test_utils import LeNet5
    from advertorch_examples.utils import get_mnist_test_loader
    from advertorch_examples.utils import TRAINED_MODEL_PATH

    # Setup
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")

    model = LeNet5()
    model_path = os.path.join(TRAINED_MODEL_PATH, model_filename)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    constraint = constopt.constraints.make_LpBall(alpha=1., p=p)

    test_loader = get_mnist_test_loader(batch_size=1, shuffle=True)
    # Just get first batch
    for data, target in test_loader:
        break

    data, target = data.to(device), target.to(device)

    adv = Adversary(data.shape, constraint, algorithm, device=device, random_init=random_init)
    optimizer = adv.optimizer
    # Logging

    store = Store(os.path.join(OUT_DIR, "test_mnist/"))
    store.add_table('metadata', {'algorithm': str,
                                 'step-size': float,
                                 'p': float,
                                 'training_mode': str,
                                 'random_init': int})

    mode = 'adv' if 'adv' in model_filename else 'cln'
    store['metadata'].append_row({'algorithm': optimizer.name,
                                  'step-size': step_size,
                                  'p': p,
                                  'training_mode': mode,
                                  'random_init': int(random_init)})

    table_name = "L" + str(int(p)) + " ball" if p != np.inf else "Linf Ball"
    store.add_table(table_name, {'func_val': float, 'FW gap': float,
                                 'norm delta': float})

    # Get nominal loss
    # output = model(data)
    # loss = criterion(output, target)
    # Run and log perturbation
    adv_loss, delta = adv.perturb(data, target, model, criterion, step_size,
                                  iterations=20,
                                  tol=1e-7, store=store)
