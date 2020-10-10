from tqdm import tqdm

import numpy as np
import torch
from torch import nn

from easydict import EasyDict

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader

import constopt
from constopt.adversary import Adversary
from constopt.optim import PGD, PGDMadry, FrankWolfe, MomentumFrankWolfe

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
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)

# Inner optimization parameters
eps = 0.3
constraint = constopt.constraints.make_LpBall(alpha=eps, p=np.inf)
inner_iter = 40
inner_iter_test = 20
step_size = 2 * eps / inner_iter  # Step size recommended in Madry's paper
# step_size = 1.25 * eps  # Step size used with random initialization
step_size_test = 2 * eps / inner_iter_test
random_init = False  # Sample the starting optimization point uniformly at random in the constraint set

# TODO: Actually log and plot stuff
# Logging
losses = []
accuracies = []
adv_losses = []
adv_accuracies = []

# adv_opt_class = PGD  # Seems like PGD doesn't work that well
# adv_opt_class = PGDMadry  # To beat
adv_opt_class = FrankWolfe  # Seems good with few steps, ie 2. Using 10 steps breaks the model.
# adv_opt_class = MomentumFrankWolfe  # Same as FW: 2 steps works nicely

# Training loop
for epoch in range(nb_epochs):
    model.train()
    train_loss = 0.
    for data, target in tqdm(train_loader, desc=f'Training epoch {epoch}/{nb_epochs - 1}'):
        data, target = data.to(device), target.to(device)

        adv = Adversary(data.shape, constraint, adv_opt_class,
                        device=device, random_init=random_init)
        _, delta = adv.perturb(data, target, model, criterion,
                               step_size,
                               iterations=inner_iter,
                               tol=1e-7)
        optimizer.zero_grad()
        adv_loss = criterion(model(data + delta), target)
        adv_loss.backward()
        optimizer.step()

        train_loss += adv_loss.item()
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
        adv_pgd = Adversary(data.shape, constraint, PGD, device=device, random_init=False)
        adv_pgd_madry = Adversary(data.shape, constraint, PGDMadry, device=device, random_init=False)
        adv_fw = Adversary(data.shape, constraint, FrankWolfe, device=device, random_init=False)
        adv_mfw = Adversary(data.shape, constraint, MomentumFrankWolfe, device=device, random_init=False)
        # Compute different perturbations
        _, delta_pgd = adv_pgd.perturb(data, target, model, criterion, step_size_test,
                               iterations=inner_iter_test,
                               tol=1e-7)
        _, delta_pgd_madry = adv_pgd_madry.perturb(data, target, model, criterion, step_size_test,
                               iterations=inner_iter_test,
                               tol=1e-7)
        _, delta_fw = adv_fw.perturb(data, target, model, criterion, step_size_test,
                               iterations=inner_iter_test,
                               tol=1e-7)
        _, delta_mfw = adv_mfw.perturb(data, target, model, criterion, step_size_test,
                               iterations=inner_iter_test,
                               tol=1e-7)
        # Compute corresponding predictions        
        _, pred = model(data).max(1)
        _, adv_pred_pgd = model(data + delta_pgd).max(1)
        _, adv_pred_pgd_madry = model(data + delta_pgd_madry).max(1)
        _, adv_pred_fw = model(data + delta_fw).max(1)
        _, adv_pred_mfw = model(data + delta_mfw).max(1)

        # Get clean accuracies
        report.nb_test += data.size(0)
        report.correct += pred.eq(target).sum().item()
        # Adversarial
        report.correct_adv_pgd += adv_pred_pgd.eq(target).sum().item()
        report.correct_adv_pgd_madry += adv_pred_pgd_madry.eq(target).sum().item()
        report.correct_adv_fw += adv_pred_fw.eq(target).sum().item()
        report.correct_adv_mfw += adv_pred_mfw.eq(target).sum().item()

    print(f'Val acc on clean examples (%): {report.correct / report.nb_test * 100.:.3f}')
    print(f'Val acc on adversarial examples PGD (%): {report.correct_adv_pgd / report.nb_test * 100.:.3f}')
    print(f'Val acc on adversarial examples PGD Madry (%): {report.correct_adv_pgd_madry / report.nb_test * 100.:.3f}')
    print(f'Val acc on adversarial examples FW (%): {report.correct_adv_fw / report.nb_test * 100.:.3f}')
    print(f'Val acc on adversarial examples MFW (%): {report.correct_adv_mfw / report.nb_test * 100.:.3f}')
