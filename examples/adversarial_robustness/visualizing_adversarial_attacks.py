"""This example shows how to generate and plot adversarial examples for a batch of datapoints from CIFAR-10,
and compares the examples from different constraint sets, penalizations and solvers."""


from itertools import product


import torch
import torchvision

from robustbench.data import load_cifar10
from robustbench.utils import load_model

import matplotlib.pyplot as plt

import chop
from chop.image import matplotlib_imshow_batch
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

# Plot loss functions per datapoint
fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
for k in range(batch_size):
    ax[k].plot([-trace[k] for trace in callback_L2.trace_f])

plt.tight_layout()
plt.savefig(f"examples/plots/adversarial_examples/adv_losses_L{constraint.p}.png")

# Plot adversarial images
fig, ax = plt.subplots(nrows=5, ncols=batch_size, figsize=(12, 10))

# Plot clean data
matplotlib_imshow_batch(data, labels=[classes[k] for k in target], axes=ax[0, :],
                        title="Original images")

# Adversarial Lp images
adv_output = model(data + delta)
adv_labels = torch.argmax(adv_output, dim=-1)
matplotlib_imshow_batch(data + delta, labels=[classes[k] for k in adv_labels], axes=ax[1, :],
                        title=f'L{constraint.p} perturbed image')

# Perturbation
matplotlib_imshow_batch(delta, axes=ax[3, :], normalize=True,
                        title=f'L{constraint.p} perturbation')


print("GroupL1 penalty.")

def group_patches(x_patch_size=8, y_patch_size=8, x_image_size=32, y_image_size=32, n_channels=3):
    groups = []
    for m in range(int(x_image_size / x_patch_size)):
        for p in range(int(y_image_size / y_patch_size)):
            groups.append([(c, m * x_patch_size + i, p * y_patch_size + j)
                           for c, i, j in product(range(n_channels),
                                                  range(x_patch_size),
                                                  range(y_patch_size))])
    return groups


alpha = 1e-1
groups = group_patches(x_patch_size=4, y_patch_size=4)
penalty = chop.penalties.GroupL1(alpha, groups)
adversary_group = chop.Adversary(chop.optim.minimize_three_split)

callback_group = Trace(callable=lambda kw: criterion(model(data + kw['x']), target))

_, delta_group = adversary_group.perturb(data, target, model, criterion,
                                         prox1=image_constraint_prox,
                                         prox2=penalty.prox,
                                         max_iter=20,
                                         callback=callback_group)

adv_output_group = model(data + delta_group)
adv_labels_group = torch.argmax(adv_output_group, dim=-1)

matplotlib_imshow_batch(data + delta_group, labels=(classes[k] for k in adv_labels_group),
                        axes=ax[2, :],
                        title='Group Lasso perturbed image')

matplotlib_imshow_batch(delta_group, axes=ax[4, :], normalize=True,
                        title='Group Lasso perturbation')

plt.tight_layout()
plt.savefig(f"examples/plots/adversarial_examples/adversarial_comparison_L{constraint.p}.png")


# Plot loss values
fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
for k in range(batch_size):
    ax[k].plot([trace[k] for trace in callback_group.trace_callable])
plt.tight_layout()
plt.savefig("examples/plots/adversarial_examples/adv_losses_groupLASSO.png")
