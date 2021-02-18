"""
Universal Adversarial Examples
================================

This example shows how to generate and plot universeral adversarial examples for CIFAR-10,
and compares the examples from different constraint sets, penalizations and solvers.

We solve the following problem:
..math:
    \max_{\delta \in \mathcal B} \frac{1}{n} \sum_{i=1}^n \ell(h_\theta(x_i+\delta), y_i) 
"""

import torch
import torchvision
# from torchvision import transforms
# from robustbench.data import load_cifar10
from robustbench.utils import load_model
from robustness.datasets import ImageNet as ImageNetRobustness
from robustness.model_utils import make_and_restore_model
import matplotlib.pyplot as plt

import chop
from chop.utils.image import group_patches, matplotlib_imshow_batch
from chop.utils.data import ImageNet, CIFAR10, NormalizingModel
from chop.utils.logging import Trace
from chop.utils import closure
from sklearn.metrics import f1_score


from tqdm import tqdm 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

batch_size = 100

# Note that this example uses load_cifar10 from the robustbench library
# data, target = load_cifar10(n_examples=batch_size, data_dir='~/datasets')


# data_dir = "/scratch/data/imagenet12/"
# dataset = ImageNet(data_dir, normalize=False)
data_dir = "/scratch/data/"
dataset = CIFAR10(data_dir, normalize=False)

loaders = dataset.loaders(batch_size, batch_size)

normalize = dataset.normalize
unnormalize = dataset.unnormalize

classes = dataset.classes


# CIFAR10 model
model = load_model('Standard')  # Can be changed to any model from the robustbench model zoo
model = NormalizingModel(model, dataset)

model = model.to(device)
# Attack criterion
criterion = torch.nn.CrossEntropyLoss()

n_epochs = 1
steps_per_batch = 2
groups = group_patches(x_patch_size=8, y_patch_size=8, x_image_size=32, y_image_size=32)
alpha = 2e-1 * len(groups)
constraint = chop.constraints.GroupL1Ball(alpha, groups)
adversary = chop.Adversary(chop.optim.minimize_frank_wolfe)

callback = Trace()

delta = torch.normal(torch.zeros(1, 3, 32, 32))
delta.requires_grad_(True)
model.eval()

for it in range(n_epochs):

    for data, target in tqdm(loaders.train):

        @closure
        def loss_fun(delta):
            pert_data = torch.where(delta == 0, data, delta)
            return -criterion(model(pert_data), target)

        algorithm = chop.optim.minimize_frank_wolfe

        algorithm(loss_fun, delta, constraint.lmo, max_iter=steps_per_batch,
                  callback=callback)

plt.plot(callback.trace_f)
plt.savefig("universal_loss.png")