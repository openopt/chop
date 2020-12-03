"""This example shows how to generate and plot adversarial examples for a batch of datapoints from cifar-10,
and compares the examples from different constraint sets, penalizations and solvers."""


from itertools import product


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
print("L1 norm constraint.")
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


_, delta = adversary.perturb(data, target, model, criterion, prox=prox, max_iter=10)

adv_img_grid = torchvision.utils.make_grid(data + delta)

matplotlib_imshow(adv_img_grid)
plt.savefig("examples/plots/adversarial_examples/adv_imgs_L1.png")


print("GroupL1 penalty.")
adversary = chop.Adversary(chop.optim.minimize_three_split)


def group_patches(row_groups=8, col_groups=8, n_rows=32, n_cols=32, n_channels=3):
    groups = []
    for m in range(row_groups):
        for p in range(col_groups):
            groups.append([(c, m * row_groups + i, p * col_groups + j)
                           for c, i, j in product(range(n_channels),
                                                  range(int(n_rows / row_groups)),
                                                  range(int(n_rows /col_groups)))])
    return groups

groups = group_patches()
constraint = chop.penalties.GroupL1(alpha, groups)
_, delta = adversary.perturb(data, target, model, criterion,
                             prox1=constraint.prox,
                             prox2=image_constraint_prox,
                             max_iter=10)

adv_img_grid = torchvision.utils.make_grid(data + delta)

matplotlib_imshow(adv_img_grid)
plt.savefig("examples/plots/adversarial_examples/adv_imgs_groupLASSO.png")
