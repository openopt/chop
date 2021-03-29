"""
Constrained neural network training.
======================================
Trains a LeNet5 model on MNIST using constraints on the weights.
This example is inspired by the official PyTorch MNIST example, which
can be found [here](https://github.com/pytorch/examples/blob/master/mnist/main.py).
"""
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
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
# Model setup


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


print("Initializing model.")
model = Net()
model.to(device)

criterion = nn.CrossEntropyLoss()

# Outer optimization parameters
nb_epochs = 20
momentum = .9
lr = 0.3

# Make constraints
print("Preparing constraints.")
constraints = chop.constraints.make_Lp_model_constraints(model, p=1, value=10000)
proxes = [constraint.prox if constraint else None for constraint in constraints]
lmos = [constraint.lmo if constraint else None for constraint in constraints]

print("Projecting model parameters in their associated constraint sets.")
chop.constraints.make_feasible(model, proxes)

optimizer = chop.stochastic.FrankWolfe(model.parameters(), lmos,
                                       lr=lr, momentum=momentum,
                                       weight_decay=3e-4,
                                       normalization='gradient')

bias_params = [param for name, param in model.named_parameters() if 'bias' in name]
bias_opt = chop.stochastic.PGD(bias_params, lr=1e-2)

print("Training...")
# Training loop
for epoch in range(nb_epochs):
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

    # Evaluate on clean and adversarial test data

    model.eval()
    report = EasyDict(nb_test=0, correct=0, correct_adv_pgd=0,
                      correct_adv_pgd_madry=0,
                      correct_adv_fw=0, correct_adv_mfw=0)

    for data, target in tqdm(loaders.test, desc=f'Val epoch {epoch}/{nb_epochs - 1}'):
        data, target = data.to(device), target.to(device)

        # Compute corresponding predictions        
        _, pred = model(data).max(1)

        # Get clean accuracies
        report.nb_test += data.size(0)
        report.correct += pred.eq(target).sum().item()

    print(f'Val acc on clean examples (%): {report.correct / report.nb_test * 100.:.3f}')
