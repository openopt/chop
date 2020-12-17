"""This example shows how to generate and plot adversarial examples for a batch of datapoints from CIFAR-10,
and compares the examples from different constraint sets, penalizations and solvers."""


from itertools import product


import torch
import torchvision

from robustbench.data import load_cifar10
from robustbench.utils import load_model

import matplotlib.pyplot as plt

import chop
from chop.image import group_patches, matplotlib_imshow_batch
from chop.logging import Trace

# Create folder for saving plots

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size = 8

# Note that this example uses load_cifar10 from the robustbench library
data, target = load_cifar10(n_examples=batch_size, data_dir='~/datasets')
data = data.to(device)
target = target.to(device)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


model = load_model('Standard')  # Can be changed to any model from the robustbench model zoo
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Define the constraint set + initial point
print("L2 norm constraint.")
alpha = .5
constraint = chop.constraints.L2Ball(alpha)


def image_constraint_prox(delta, step_size=None):
    adv_img = torch.clamp(data + delta, 0, 1)
    delta = adv_img - data
    return delta


def prox(delta, step_size=None):
    delta = constraint.prox(delta, step_size)
    delta = image_constraint_prox(delta, step_size)
    return delta


adversary = chop.Adversary(chop.optim.minimize_pgd_madry)
callback_L2 = Trace()
_, delta = adversary.perturb(data, target, model, criterion,
                             prox=prox,
                             lmo=constraint.lmo,
                             max_iter=20,
                             step=2. / 20,
                             callback=callback_L2)

# Plot adversarial images
fig, ax = plt.subplots(nrows=7, ncols=batch_size, figsize=(16, 14))

# Plot clean data
matplotlib_imshow_batch(data, labels=[classes[k] for k in target], axes=ax[0, :],
                        title="Original images")

# Adversarial Lp images
adv_output = model(data + delta)
adv_labels = torch.argmax(adv_output, dim=-1)
matplotlib_imshow_batch(data + delta, labels=[classes[k] for k in adv_labels], axes=ax[1, :],
                        title=f'L{constraint.p}')

# Perturbation
matplotlib_imshow_batch(abs(delta), axes=ax[4, :], normalize=True,
                        title=f'L{constraint.p}')


print("GroupL1 constraint.")

groups = group_patches(x_patch_size=8, y_patch_size=8)
alpha = .5 * len(groups)
constraint_group = chop.constraints.GroupL1Ball(alpha, groups)
adversary_group = chop.Adversary(chop.optim.minimize_frank_wolfe)

# callback_group = Trace(callable=lambda kw: criterion(model(data + kw['x']), target))
callback_group = Trace()

_, delta_group = adversary_group.perturb(data, target, model, criterion,
                                         lmo=constraint_group.lmo,
                                         max_iter=20,
                                         callback=callback_group)
delta_group = image_constraint_prox(delta_group)

# Show adversarial examples and perturbations
adv_output_group = model(data + delta_group)
adv_labels_group = torch.argmax(adv_output_group, dim=-1)

matplotlib_imshow_batch(data + delta_group, labels=(classes[k] for k in adv_labels_group),
                        axes=ax[2, :],
                        title='Group Lasso')

matplotlib_imshow_batch(abs(delta_group), axes=ax[5, :], normalize=True,
                        title='Group Lasso')


print("Nuclear norm ball adv examples")

alpha = 3
constraint_nuc = chop.constraints.NuclearNormBall(alpha)


def prox_nuc(delta, step_size=None):
    delta = constraint_nuc.prox(delta, step_size)
    delta = image_constraint_prox(delta, step_size)
    return delta


adversary = chop.Adversary(chop.optim.minimize_frank_wolfe)
callback_nuc = Trace()
_, delta_nuc = adversary.perturb(data, target, model, criterion,
                                #  prox=prox,
                                 lmo=constraint_nuc.lmo,
                                 max_iter=20,
                                #  step=2. / 20,
                                 callback=callback_nuc)

# Clamp last iterate to image space
delta_nuc = image_constraint_prox(delta_nuc)

# Add nuclear examples to plot
adv_output_nuc = model(data + delta_nuc)
adv_labels_nuc = torch.argmax(adv_output_nuc, dim=-1)

matplotlib_imshow_batch(data + delta_nuc, labels=(classes[k] for k in adv_labels_nuc),
                        axes=ax[3, :],
                        title='Nuclear Norm')

matplotlib_imshow_batch(abs(delta_nuc), axes=ax[6, :], normalize=True,
                        title='Nuclear Norm')


plt.tight_layout()
plt.show()


# TODO refactor this in functions

# Plot group lasso loss values
fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
for k in range(batch_size):
    ax[k].plot([trace[k] for trace in callback_group.trace_f])
plt.tight_layout()
plt.show()

# Plot loss functions per datapoint
fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
for k in range(batch_size):
    ax[k].plot([-trace[k] for trace in callback_nuc.trace_f])

plt.tight_layout()
plt.show()

# Plot loss functions per datapoint
fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
for k in range(batch_size):
    ax[k].plot([-trace[k] for trace in callback_L2.trace_f])

plt.tight_layout()
plt.show()
