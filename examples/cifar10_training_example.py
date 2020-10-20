import os

from tqdm import tqdm

import numpy as np
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict

from torchvision.models import resnet18

import constopt
from constopt.adversary import Adversary
from constopt.optim import PGD, PGDMadry, FrankWolfe, MomentumFrankWolfe
from constopt.data_utils import ld_cifar10

# Setup
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data Loaders
loader = ld_cifar10()
train_loader = loader.train
test_loader = loader.test

# Model setup
model = resnet18()
model.to(device)
criterion = nn.CrossEntropyLoss()

# TODO use SOTA schedulers etc...
# Outer optimization parameters
nb_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Inner optimization parameters
# eps = 8. / 255  # From Madry's paper
eps = 1.
constraint = constopt.constraints.make_LpBall(alpha=eps, p=np.inf)
inner_iter = 4
inner_iter_test = 20
step_size = 2 * eps / inner_iter  # Step size recommended in Madry's paper
# step_size = 1.25 * eps  # Step size used with random initialization
step_size_test = 2 * eps / inner_iter_test
random_init = False  # Sample the starting optimization point uniformly at random in the constraint set


# adv_opt_class = PGD  # Seems like PGD doesn't work that well
# adv_opt_class = PGDMadry  # To beat
adv_opt_class = FrankWolfe  # Seems good with few steps, ie 2. Using 10 steps breaks the model.
# adv_opt_class = MomentumFrankWolfe  # Same as FW: 2 steps works nicely


# TODO: Actually log and plot stuff
# Logging
writer = SummaryWriter(os.path.join("logging/cifar10/", adv_opt_class.name))

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
    writer.add_scalar("Loss/train", train_loss, epoch)
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

    report.correct /= report.nb_test
    report.correct_adv_pgd /= report.nb_test
    report.correct_adv_pgd_madry /= report.nb_test
    report.correct_adv_fw /= report.nb_test
    report.correct_adv_mfw /= report.nb_test

    print(f'Val acc on clean examples (%): {report.correct * 100.:.3f}')
    print(f'Val acc on adversarial examples PGD (%): {report.correct_adv_pgd * 100.:.3f}')
    print(f'Val acc on adversarial examples PGD Madry (%): {report.correct_adv_pgd_madry * 100.:.3f}')
    print(f'Val acc on adversarial examples FW (%): {report.correct_adv_fw * 100.:.3f}')
    print(f'Val acc on adversarial examples MFW (%): {report.correct_adv_mfw * 100.:.3f}')
    
    writer.add_scalars("Test/Adv", report, epoch)
