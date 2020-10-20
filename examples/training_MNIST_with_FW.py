"""Trains a LeNet5 model on MNIST using constraints on the weights.
"""
from tqdm import tqdm

import numpy as np
import torch
from torch import nn

from easydict import EasyDict

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader

import constopt

# Setup
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data Loaders
train_loader = get_mnist_train_loader(batch_size=128, shuffle=True)
test_loader = get_mnist_test_loader(batch_size=512, shuffle=True)

# Model setup
model = LeNet5()
model.to(device)
criterion = nn.CrossEntropyLoss()

# Outer optimization parameters
nb_epochs = 20
step_size = .1
momentum = .9

# TODO: tune hyperparams for algorithms, and constraint sets
# Choose optimizer here
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
alpha = 1.
constraint = constopt.constraints.LinfBall(alpha)
# optimizer = constopt.optim.PGD(model.parameters(), constraint)
# optimizer = constopt.optim.PGDMadry(model.parameters(), constraint)
# optimizer = constopt.optim.FrankWolfe(model.parameters(), constraint)
# optimizer = constopt.optim.MomentumFrankWolfe(model.parameters(), constraint)

# TODO: Actually log and plot stuff
# Logging
losses = []
accuracies = []
adv_losses = []
adv_accuracies = []


# Training loop
for epoch in range(nb_epochs):
    model.train()
    train_loss = 0.
    for data, target in tqdm(train_loader, desc=f'Training epoch {epoch}/{nb_epochs - 1}'):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step(step_size=step_size, momentum=momentum)

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
