import numpy as np
import torch
from torch import nn

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader

import constopt
from constopt.adversary import Adversary
from constopt.optim import PGD

# Setup
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data Loaders
train_loader = get_mnist_test_loader(batch_size=128, shuffle=True)
test_loader = get_mnist_test_loader(batch_size=512, shuffle=True)

# Model setup
model = LeNet5()
model.to(device)
criterion = nn.CrossEntropyLoss()

# Outer optimization parameters
nb_epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)

# Inner optimization parameters
eps = 1.
constraint = constopt.constraints.make_LpBall(alpha=eps, p=np.inf)
inner_iter = 20
step_size = 2 * eps / inner_iter

# Logging
losses = []
accuracies = []
adv_losses = []
adv_accuracies = []

# Training loop
for epoch in range(nb_epochs):
    model.train()
    train_loss = 0.
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        adv = Adversary(data.shape, constraint, PGD, device=device, random_init=False)
        _, delta = adv.perturb(data, target, model, criterion, step_size,
                               iterations=inner_iter,
                               tol=1e-7)

        optimizer.zero_grad()
        adv_loss = criterion(model(data + delta), target)
        adv_loss.backward()
        optimizer.step()

        train_loss += adv_loss
    train_loss /= len(train_loader)
    print(f'epoch: {epoch}/{nb_epochs}, train loss: {train_loss:.3f}')

    # Evaluate on clean and adversarial test data
    
    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        
        