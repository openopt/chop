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
from chop import utils
from chop.logging import Trace


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size = 8

# Note that this example uses load_cifar10 from the robustbench library
data, target = load_cifar10(n_examples=batch_size, data_dir='~/datasets')
data = data.to(device)
target = target.to(device)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

print("Plotting clean images")
img_grid = torchvision.utils.make_grid(data, normalize=True, range=(0, 1))
matplotlib_imshow(img_grid)
plt.savefig("examples/plots/adversarial_examples/cln_imgs.png")


model = load_model('Standard')  # Can be changed to any model from the robustbench model zoo
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Define the constraint set + initial point
print("L1 norm constraint.")
alpha = 8 / 255.
constraint = chop.constraints.LinfBall(alpha)


def image_constraint_prox(delta, step_size=None):
    adv_img = torch.clamp(data + delta, 0, 1)
    delta = adv_img - data
    return delta


def prox(delta, step_size=None):
    delta = constraint.prox(delta, step_size)
    delta = image_constraint_prox(delta, step_size)
    return delta


adversary = chop.Adversary(chop.optim.minimize_pgd_madry)
callback_L1 = Trace()
_, delta = adversary.perturb(data, target, model, criterion,
                             prox=prox, lmo=constraint.lmo,
                             max_iter=20,
                             step=2.5 / 20, callback=callback_L1)


fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
for k in range(batch_size):
    ax[k].plot([trace[k] for trace in callback_L1.trace_f])

plt.tight_layout()
plt.savefig("examples/plots/adversarial_examples/adv_losses_Linf.png")

plt.figure()
adv_img_grid = torchvision.utils.make_grid(data + delta)
matplotlib_imshow(adv_img_grid)
plt.savefig("examples/plots/adversarial_examples/adv_imgs_Linf.png")

plt.figure()
delta_grid = torchvision.utils.make_grid(delta, normalize=True)
matplotlib_imshow(delta_grid)
plt.savefig("examples/plots/adversarial_examples/perturbation_Linf.png")

print("GroupL1 penalty.")
adversary = chop.Adversary(chop.optim.minimize_three_split)


def group_patches(x_patch_size=8, y_patch_size=8, x_image_size=32, y_image_size=32, n_channels=3):
    groups = []
    for m in range(int(x_image_size / x_patch_size)):
        for p in range(int(y_image_size / y_patch_size)):
            groups.append([(c, m * x_patch_size + i, p * y_patch_size + j)
                           for c, i, j in product(range(n_channels),
                                                  range(x_patch_size),
                                                  range(y_patch_size))])
    return groups


alpha = 1e1
groups = group_patches()
penalty = chop.penalties.GroupL1(alpha, groups)

callback = Trace()

_, delta = adversary.perturb(data, target, model, criterion,
                             prox1=penalty.prox,
                             prox2=image_constraint_prox,
                             max_iter=20, callback=callback)

fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
for k in range(batch_size):
    ax[k].plot([trace[k] for trace in callback.trace_f])
plt.tight_layout()
plt.savefig("examples/plots/adversarial_examples/adv_losses_groupLASSO.png")

fig = plt.figure()
adv_img_grid = torchvision.utils.make_grid(data + delta, normalize=True)
matplotlib_imshow(adv_img_grid)
plt.savefig("examples/plots/adversarial_examples/adv_imgs_groupLASSO.png")

plt.figure()
delta_grid = torchvision.utils.make_grid(delta, normalize=True)
matplotlib_imshow(delta_grid)
plt.savefig("examples/plots/adversarial_examples/perturbations_groupLASSO.png")