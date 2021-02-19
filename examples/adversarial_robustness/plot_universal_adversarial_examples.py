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
from robustbench.utils import load_model
import matplotlib.pyplot as plt

import chop
from chop.utils.image import group_patches, matplotlib_imshow_batch, matplotlib_imshow
from chop.utils.data import ImageNet, CIFAR10, NormalizingModel
from chop.utils.logging import Trace
from tqdm import tqdm 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100

data_dir = "~/datasets/"
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
steps_per_batch = 5
groups = group_patches(x_patch_size=8, y_patch_size=8, x_image_size=32, y_image_size=32)
alpha = 2e-1 * len(groups)

callback_delta = Trace()
callback_rho = Trace()

delta = torch.zeros(3, 32, 32)
delta = delta.to(device)
delta.requires_grad_(True)

rho = torch.tensor([0.5]).to(device)
rho.requires_grad_(True)

constraint = chop.constraints.L1Ball(100.)
delta_opt = chop.stochastic.PGDMadry([delta], constraint, lr=.05)
rho_constraint = chop.constraints.Simplex(.5)  # rho \in [0,1]
rho_opt = chop.stochastic.PGDMadry([rho], rho_constraint, lr=.05)
model.eval()

losses = []


for it in range(n_epochs):

    for data, target in tqdm(loaders.train):
        data = data.to(device)
        target = target.to(device)

        def loss_fun(delta, rho):
            pert_data = data + rho * (delta - data)
            return -criterion(model(pert_data), target)

        delta_opt.zero_grad()
        rho_opt.zero_grad()

        loss_val = loss_fun(delta, rho)
        loss_val.backward()

        delta_opt.step()
        with torch.no_grad():
            delta = torch.clamp(delta, 0, 1)
        rho_opt.step()

        losses.append(-loss_val.item())

print(f"Optimal transparency (rho) is {rho}")

plt.plot(losses)

fig, ax = plt.subplots()
matplotlib_imshow(delta)


fig, ax = plt.subplots()
data = data.to(device)
pert_image = data[0] + rho * (delta - data[0])
matplotlib_imshow(pert_image)