"""
Visualizing Adversarial Examples
================================

This example shows how to generate and plot adversarial examples for a batch of datapoints from CIFAR-10,
and compares the examples from different constraint sets, penalizations and solvers.

"""

# TODO: REFACTOR DATASETS FROM OUR DATALOADING UTILITIES

import torch
import torchvision
# from torchvision import transforms
# from robustbench.data import load_cifar10
# from robustbench.utils import load_model

import matplotlib.pyplot as plt

import chop
from chop.utils.image import group_patches, matplotlib_imshow_batch
from chop.utils.data import ImageNet, CIFAR10, NormalizingModel
from chop.utils.logging import Trace

from sklearn.metrics import f1_score

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

batch_size = 4

# Note that this example uses load_cifar10 from the robustbench library
# data, target = load_cifar10(n_examples=batch_size, data_dir='~/datasets')

data_dir = "/scratch/data/imagenet12/"
dataset = ImageNet(data_dir, normalize=False)

# data_dir = "~/datasets/"
# dataset = CIFAR10()

normalize = dataset.normalize
unnormalize = dataset.unnormalize

data, target = dataset.load_k(batch_size, train=True, device=device,
                              shuffle=True)
classes = dataset.classes

# ImageNet model
model = torchvision.models.resnet18(pretrained=True)

# CIFAR10 model
# model = load_model('Standard')  # Can be changed to any model from the robustbench model zoo

model = model.to(device)
# Attack criterion
criterion = torch.nn.CrossEntropyLoss(reduction='none')


# Add first layer normalization
# since the data is not normalized on loading.
model = NormalizingModel(model, dataset)

# Define the constraint set + initial point
print("L2 norm constraint.")
alpha = 3
constraint = chop.constraints.L2Ball(alpha)

max_str_length = 15 # for plot: max length of class

def image_constraint_prox(delta, step_size=None):
    adv_img = torch.clamp(data + delta, 0, 1)
    delta = adv_img - data
    return delta


# TODO: think above using Dykstra instead of 2 alternate projections
def prox(delta, step_size=None):
    """This needs to clip the data renormalized to [0, 1].
    The epsilon scale is w/ regard to this unit box constraint."""

    delta = constraint.prox(delta, step_size)
    delta = image_constraint_prox(delta, step_size)
    return delta


adversary = chop.Adversary(chop.optim.minimize_pgd_madry)
callback_L2 = Trace()

# perturb's iterates are delta in [-mean / std, (1 - mean)/ std]
_, delta = adversary.perturb(data, target, model, criterion,
                             prox=prox,
                             lmo=constraint.lmo,
                             max_iter=20,
                             step=2. / 20,
                             callback=callback_L2)

# Plot adversarial images
fig, ax = plt.subplots(ncols=8, nrows=batch_size, figsize=(12, 6))

# Plot clean data
matplotlib_imshow_batch(data, labels=[classes[int(k)][:max_str_length] for k in target], axes=ax[:, 0],
                        title="Ground Truth")

preds = model(data).argmax(dim=-1)
matplotlib_imshow_batch(data, labels=[classes[int(k)][:max_str_length] for k in preds], axes=ax[:, 1],
                        title="Prediction")

# Adversarial Lp images
adv_output = model(data + delta)
adv_labels = torch.argmax(adv_output, dim=-1)
matplotlib_imshow_batch(data + delta, labels=[classes[int(k)][:max_str_length] for k in adv_labels], axes=ax[:, 2],
                        title=f'L{constraint.p}')

# Perturbation
matplotlib_imshow_batch(abs(delta), axes=ax[:, 5], normalize=True,
                        # title=f'L{constraint.p}', negative=True)
                        title=f'L{constraint.p}', negative=False)


print("GroupL1 constraint.")

# CIFAR-10
# groups = group_patches(x_patch_size=8, y_patch_size=8, x_image_size=32, y_image_size=32)
# Imagenet
groups = group_patches(x_patch_size=28, y_patch_size=28, x_image_size=224, y_image_size=224)

for eps in [5e-2]:
    alpha = eps * len(groups)
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

    matplotlib_imshow_batch(data + delta_group, labels=(classes[int(k)][:max_str_length] for k in adv_labels_group),
                            axes=ax[:, 3],
                            title='Group Lasso')

    matplotlib_imshow_batch(abs(delta_group), axes=ax[:, 6], normalize=True,
                            # title='Group Lasso', negative=True)
                            title='Group Lasso', negative=False)

    print(f"F1 score: {f1_score(target.detach().cpu(), adv_labels_group.detach().cpu(), average='macro'):.3f}"
          f" for alpha={alpha:.4f}")

print("Nuclear norm ball adv examples")

for alpha in [.5]:
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

    matplotlib_imshow_batch(data + delta_nuc, labels=(classes[int(k)][:max_str_length] for k in adv_labels_nuc),
                            axes=ax[:, 4],
                            title='Nuclear Norm')

    matplotlib_imshow_batch(abs(delta_nuc), axes=ax[:, 7], normalize=True,
                            # title='Nuclear Norm', negative=True)
                            title='Nuclear Norm', negative=False)

    print(f"F1 score: {f1_score(target.detach().cpu(), adv_labels_nuc.detach().cpu(), average='macro'):.3f}"
          f" for alpha={alpha:.4f}")

plt.subplots_adjust(bottom=.06, top=0.5)

plt.tight_layout()
plt.show()


# TODO refactor this in functions

# Plot group lasso loss values
# fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
# for k in range(batch_size):
#     ax[k].plot([-trace[k] for trace in callback_group.trace_f])
# plt.tight_layout()
# plt.show()

# # Plot loss functions per datapoint
# fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
# for k in range(batch_size):
#     ax[k].plot([-trace[k] for trace in callback_nuc.trace_f])

# plt.tight_layout()
# plt.show()

# # Plot loss functions per datapoint
# fig, ax = plt.subplots(figsize=(6, 10), nrows=batch_size, sharex=True)
# for k in range(batch_size):
#     ax[k].plot([-trace[k] for trace in callback_L2.trace_f])

# plt.tight_layout()
# plt.show()
