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
dataset = chop.utils.data.MNIST("~/datasets/")
loaders = dataset.loaders()
# print("Loading data...")
# dataset = chop.utils.data.CIFAR10("~/datasets/")
# loaders = dataset.loaders()
# Model setup


print("Initializing model.")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


model = Net()
model.to(device)

criterion = nn.CrossEntropyLoss()

# Outer optimization parameters
nb_epochs = 200
momentum = .9
lr_lmo = 'sublinear'
lr_prox = 'sublinear'

# Make constraints
print("Preparing constraints.")
constraints_sparsity = chop.constraints.make_model_constraints(model,
                                                               ord=1,
                                                               value=10000,
                                                               constrain_bias=False)
constraints_low_rank = chop.constraints.make_model_constraints(model,
                                                               ord='nuc',
                                                               value=1000,
                                                               constrain_bias=False)
proxes = [constraint.prox if constraint else None
          for constraint in constraints_sparsity]
lmos = [constraint.lmo if constraint else None
        for constraint in constraints_low_rank]

proxes_lr = [constraint.prox if constraint else None
             for constraint in constraints_low_rank]

print("Projecting model parameters in their associated constraint sets.")
chop.constraints.make_feasible(model, proxes)
chop.constraints.make_feasible(model, proxes_lr)

optimizer = chop.stochastic.SplittingProxFW(model.parameters(), lmos,
                                            proxes,
                                            lr_lmo=lr_lmo, lr_prox=lr_prox,
                                            momentum=momentum,
                                            weight_decay=0,
                                            normalization='none')

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

bias_params = (param for name, param in model.named_parameters() if chop.constraints.is_bias(name, param))
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
    # scheduler.step(val_loss)
    bias_scheduler.step(val_loss)
