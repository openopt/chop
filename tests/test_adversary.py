"""Testing our adversarial attacks"""
import pytest
import shutil
import torch
from torch import nn
import numpy as np
from cox.store import Store

import constopt
from constopt import optim
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


@pytest.mark.parametrize('algorithm', [optim.PGD, optim.PGDMadry,
                                       optim.FrankWolfe, optim.MomentumFrankWolfe])
@pytest.mark.parametrize('step_size', [1, .5, .1, .05, .001, 0.])
@pytest.mark.parametrize('p', [1, 2, np.inf])
def test_adversary(algorithm, step_size, p):
    # Setup
    torch.manual_seed(0)

    data = torch.rand((1, 25, 25))
    target = torch.zeros(1).long()

    model = LinearModel()
    criterion = nn.CrossEntropyLoss()
    constraint = constopt.constraints.make_LpBall(alpha=1., p=p)

    adv = Adversary(data.shape, constraint, algorithm)
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
