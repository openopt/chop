"""
Universal Adversarial Examples
================================

This example shows how to generate and plot universal adversarial examples for
CIFAR-10.

We solve the following problem:

..math:
    \max_{\delta \in \mathcal B} \frac{1}{n} \sum_{i=1}^n \ell(h_\theta(x_i + rho(\delta - x_i)), y_i) 
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from robustbench.utils import load_model

import chop
from chop.utils.image import matplotlib_imshow
from chop.utils.data import CIFAR10, NormalizingModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_dir = "~/datasets/"
dataset = CIFAR10(data_dir, normalize=False)

classes = dataset.classes

# CIFAR10 model
model = load_model('Standard')  # Can be changed to any model from the robustbench model zoo

# Add an initial layer to normalize data.
# This allows us to use the [0, 1] image constraint set
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
                                    prox=chop.constraints.Box(0, 1).prox,
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
plt.show()


fig, ax = plt.subplots()
matplotlib_imshow(best_delta)
plt.show()


fig, ax = plt.subplots()
data = data.to(device)
pert_image = apply_perturbation(data[0], best_delta)
matplotlib_imshow(pert_image)
plt.show()
