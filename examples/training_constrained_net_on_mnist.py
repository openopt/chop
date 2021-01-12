"""
Constrained neural network training.
======================================
Trains a LeNet5 model on MNIST using constraints on the weights.
"""
from tqdm import tqdm

import numpy as np
import torch
from torch import nn

from easydict import EasyDict

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader

import chop

# Setup
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data Loaders
train_loader = get_mnist_train_loader(batch_size=50, shuffle=True)
test_loader = get_mnist_test_loader(batch_size=512, shuffle=True)

# Model setup
model = LeNet5()
model.to(device)

criterion = nn.CrossEntropyLoss()

# Outer optimization parameters
nb_epochs = 20
momentum = .9
lr = 0.3

# Choose optimizer here
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Make constraints

alpha = 1.
constraint = chop.constraints.LinfBall(alpha)

# Project model parameters in the constraint set.
constraint.make_feasible(model)

optimizer = chop.stochastic.FrankWolfe(model.parameters(), constraint, lr=lr, momentum=momentum)

# Training loop
for epoch in range(nb_epochs):
    model.train()
    train_loss = 0.
    for data, target in tqdm(train_loader, desc=f'Training epoch {epoch}/{nb_epochs - 1}'):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f'Training loss: {train_loss:.3f}')
    # TODO: get accuracy

    # Evaluate on clean and adversarial test data

    model.eval()
    report = EasyDict(nb_test=0, correct=0, correct_adv_pgd=0,
                      correct_adv_pgd_madry=0,
                      correct_adv_fw=0, correct_adv_mfw=0)

    for data, target in tqdm(test_loader, desc=f'Val epoch {epoch}/{nb_epochs - 1}'):
        data, target = data.to(device), target.to(device)

        # Compute corresponding predictions        
        _, pred = model(data).max(1)

        # Get clean accuracies
        report.nb_test += data.size(0)
        report.correct += pred.eq(target).sum().item()

    print(f'Val acc on clean examples (%): {report.correct / report.nb_test * 100.:.3f}')
