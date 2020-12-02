"""This example shows how to generate and plot adversarial examples for a batch of datapoints from cifar-10,
and compares the examples from different constraint sets and solvers."""

import torch
import torchvision

from robustbench.data import load_cifar10
from robustbench.utils import load_model

import matplotlib.pyplot as plt

import chop
from chop.image import matplotlib_imshow

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size = 8

# Note that this example uses load_cifar10 from the robustbench library
data, target = load_cifar10(n_examples=batch_size, data_dir='~/datasets')
data = data.to(device)
target = target.to(device)


print("Plotting clean images")
img_grid = torchvision.utils.make_grid(data)
matplotlib_imshow(img_grid)
plt.savefig("examples/plots/adversarial_examples/cln_imgs.png")


model = load_model('Standard')  # Can be changed to any model from the robustbench model zoo
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Define the constraint set + initial point
alpha = 10.
constraint = chop.constraints.L1Ball(alpha)

adversary = chop.Adversary(chop.optim.minimize_pgd)


def image_constraint_prox(delta, step_size=None):
    adv_img = torch.clamp(data + delta, 0, 1)
    delta = adv_img - data
    return delta


def prox(delta, step_size=None):
    delta = constraint.prox(delta, step_size)
    delta = image_constraint_prox(delta, step_size)
    return delta


print("Perturbing the dataset.")
_, delta = adversary.perturb(data, target, model, criterion, prox=prox, max_iter=10)

print("Plotting...")

adv_img_grid = torchvision.utils.make_grid(data + delta)

matplotlib_imshow(adv_img_grid)
plt.savefig("examples/plots/adversarial_examples/adv_imgs.png")
