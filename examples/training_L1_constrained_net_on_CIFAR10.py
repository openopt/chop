"""
Constrained Neural Network Training.
======================================
Trains a ResNet model on CIFAR10 using constraints on the weights.
This example is inspired by the official PyTorch MNIST example, which
can be found [here](https://github.com/pytorch/examples/blob/master/mnist/main.py).
"""
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F
from easydict import EasyDict

import chop

# Setup
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data Loaders
print("Loading data...")
dataset = chop.utils.data.CIFAR10("~/datasets/")
loaders = dataset.loaders()
# Model setup


print("Initializing model.")
model = models.resnet18()
model.to(device)

criterion = nn.CrossEntropyLoss()

# Outer optimization parameters
nb_epochs = 200
momentum = .9
lr = 0.1

# Make constraints
print("Preparing constraints.")
constraints = chop.constraints.make_Lp_model_constraints(model, p=1, value=10000)
proxes = [constraint.prox if constraint else None for constraint in constraints]
lmos = [constraint.lmo if constraint else None for constraint in constraints]

print("Projecting model parameters in their associated constraint sets.")
chop.constraints.make_feasible(model, proxes)

optimizer = chop.stochastic.FrankWolfe(model.parameters(), lmos,
                                       lr=lr, momentum=momentum,
                                       weight_decay=5e-4,
                                       normalization='gradient')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

bias_params = [param for name, param in model.named_parameters() if 'bias' in name]
bias_opt = chop.stochastic.PGD(bias_params, lr=1e-1)
bias_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(bias_opt)


def train():
    model.train()
    train_loss = 0.
    for data, target in tqdm(loaders.train, desc=f'Training epoch {epoch}/{nb_epochs - 1}'):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        bias_opt.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        bias_opt.step()

        train_loss += loss.item()
    train_loss /= len(loaders.train)
    print(f'Training loss: {train_loss:.3f}')
    return train_loss


def eval():
    model.eval()
    report = EasyDict(nb_test=0, correct=0, correct_adv_pgd=0,
                      correct_adv_pgd_madry=0,
                      correct_adv_fw=0, correct_adv_mfw=0)
    val_loss = 0
    with torch.no_grad():
        for data, target in tqdm(loaders.test, desc=f'Val epoch {epoch}/{nb_epochs - 1}'):
            data, target = data.to(device), target.to(device)

            # Compute corresponding predictions
            logits = model(data)
            _, pred = logits.max(1)
            val_loss += criterion(logits, target)
            # Get clean accuracies
            report.nb_test += data.size(0)
            report.correct += pred.eq(target).sum().item()

    val_loss /= report.nb_test
    print(f'Val acc on clean examples (%): {report.correct / report.nb_test * 100.:.3f}')
    return val_loss


print("Training...")
# Training loop
for epoch in range(nb_epochs):
    # Evaluate on clean and adversarial test data
    train()
    val_loss = eval()
    scheduler.step(val_loss)
    bias_scheduler.step(val_loss)
