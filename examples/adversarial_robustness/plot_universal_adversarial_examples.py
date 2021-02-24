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
import numpy as np

import chop
from chop.utils.image import group_patches, matplotlib_imshow_batch, matplotlib_imshow
from chop.utils.data import ImageNet, CIFAR10, NormalizingModel
from chop.utils.logging import Trace
from tqdm import tqdm 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_dir = "~/datasets/"
dataset = CIFAR10(data_dir, normalize=False)


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
restarts = 5
batch_size = 250

loaders = dataset.loaders(batch_size, batch_size)

# Optimize freely over a patch in the image
length = 8
x_start = 12
y_start = 12
rho = torch.zeros(3, 32, 32).to(device)
rho[:, x_start:x_start+length, y_start:y_start+length] += 1.

model.eval()

losses = []
test_acc = []
test_acc_adv = []


def apply_perturbation(data, delta):
    return data + rho * (delta - data)

best_loss = -np.inf

for _ in range(restarts):
    # Random initialization
    delta = torch.rand(3, 32, 32).to(device)
    delta.requires_grad_(True)
    delta_opt = chop.stochastic.PGD([delta],
                                    constraint=chop.constraints.Box(0, 1),
                                    lr=.2, normalization='Linf')
    
    for it in range(n_epochs):
        for data, target in tqdm(loaders.train):
            data = data.to(device)
            target = target.to(device)

            def loss_fun(delta):
                adv_data = apply_perturbation(data, delta)
                return -criterion(model(adv_data), target)

            delta_opt.zero_grad()
            loss_val = loss_fun(delta)
            loss_val.backward()
            delta_opt.step()
            losses.append(-loss_val.item())

            if -loss_val.item() > best_loss:
                best_loss = -loss_val.item()
                best_delta = delta.detach().clone()

correct = 0
correct_adv = 0
n_datapoints = 0
for data, target in tqdm(loaders.test):
    n_datapoints += len(data)
    data = data.to(device)
    target = target.to(device)

    preds = model(data).argmax(1)
    adv_image = apply_perturbation(data, best_delta)
    preds_adv = model(adv_image).argmax(1)

    correct += (preds == target).sum().item()
    correct_adv += (preds_adv == target).sum().item()

correct /= n_datapoints
correct_adv /= n_datapoints

print(f"Clean accuracy: {100 * correct:.2f}")
print(f"Best attack accuracy {100 * correct_adv:.2f}")
        
plt.plot(losses, label='Training loss')
plt.legend()
plt.savefig("examples/plots/adversarial_examples/universal_losses.png")


fig, ax = plt.subplots()
matplotlib_imshow(delta)
plt.savefig("examples/plots/adversarial_examples/universal_perturbation.png")


fig, ax = plt.subplots()
data = data.to(device)
pert_image = apply_perturbation(data[0], delta)
matplotlib_imshow(pert_image)
plt.savefig("examples/plots/adversarial_examples/universal_perturbation_on_image.png")
