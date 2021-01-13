"""
Example of robust training on CIFAR10.
=========================================
"""
import matplotlib.pyplot as plt
from chop.adversary import Adversary
import torch
from tqdm import tqdm
from easydict import EasyDict

import chop

from torch.optim import SGD

from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

n_epochs = 100
batch_size = 128
batch_size_test = 100

loaders = chop.data.load_cifar10(train_batch_size=batch_size,
                                 test_batch_size=batch_size_test,
                                 data_dir='~/datasets',
                                 augment_train=True)

trainloader, testloader = loaders.train, loaders.test
n_train = len(trainloader.dataset)
n_test = len(testloader.dataset)

model = models.resnet18(pretrained=False)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Define the perturbation constraint set
max_iter_train = 7
max_iter_test = 20
alpha = 8. / 255
constraint = chop.constraints.LinfBall(alpha)
criterion_adv = torch.nn.CrossEntropyLoss(reduction='none')

print(f"Training on L{constraint.p} ball({alpha}).")


adversary = Adversary(chop.optim.minimize_pgd_madry)

results = EasyDict(train_acc=[], test_acc=[],
                   train_acc_adv=[], test_acc_adv=[],
                   train_adv_loss=[],
                   test_adv_loss=[])

for _ in range(n_epochs):

    # Train
    n_correct = 0
    n_correct_adv = 0

    model.train()

    for k, (data, target) in enumerate(trainloader):
        data = data.to(device)
        target = target.to(device)

        @torch.no_grad()
        def image_constraint_prox(delta, step_size=None):
            """Projects perturbation delta
            so that 0. <= data + delta <= 1."""

            adv_img = torch.clamp(data + delta, 0, 1)
            delta = adv_img - data
            return delta

        @torch.no_grad()
        def prox(delta, step_size=None):
            delta = constraint.prox(delta, step_size)
            delta = image_constraint_prox(delta, step_size)
            return delta

        _, delta = adversary.perturb(data, target, model,
                                     criterion_adv,
                                     prox=prox,
                                     lmo=constraint.lmo,
                                     step=2. / max_iter_train,
                                     max_iter=max_iter_train)

        optimizer.zero_grad()
        
        output = model(data)
        output_adv = model(data + delta)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        pred = torch.argmax(output, dim=-1)
        pred_adv = torch.argmax(output_adv, dim=-1)

        n_correct += (pred == target).sum().item()
        n_correct_adv += (pred_adv == target).sum().item()

    results.train_acc.append(100. * n_correct / n_train)
    results.train_acc_adv.append(100. * n_correct_adv / n_train)
    print(f"Train Accuracy: {results.train_acc[-1] :.1f}%")
    print(f"Train Adv Accuracy: {results.train_acc_adv[-1]:.1f}%")

    # Test
    n_correct = 0
    n_correct_adv = 0

    model.eval()

    for k, (data, target) in enumerate(testloader):
        data = data.to(device)
        target = target.to(device)

        @torch.no_grad()
        def image_constraint_prox(delta, step_size=None):
            """Projects perturbation delta
            so that 0. <= data + delta <= 1."""

            adv_img = torch.clamp(data + delta, 0, 1)
            delta = adv_img - data
            return delta

        @torch.no_grad()
        def prox(delta, step_size=None):
            delta = constraint.prox(delta, step_size)
            delta = image_constraint_prox(delta, step_size)
            return delta

        _, delta = adversary.perturb(data, target, model,
                                        criterion_adv,
                                        prox=prox,
                                        lmo=constraint.lmo,
                                        step=2. / max_iter_test,
                                        max_iter=max_iter_test)

        with torch.no_grad():
            output = model(data)
            output_adv = model(data + delta)

            pred = torch.argmax(output, dim=-1)
            pred_adv = torch.argmax(output_adv, dim=-1)

        n_correct += (pred == target).sum().item()
        n_correct_adv += (pred_adv == target).sum().item()

    results.test_acc.append(100. * n_correct / n_test)
    results.test_acc_adv.append(100. * n_correct_adv / n_test)

    print(f"Test Accuracy: {results.test_acc[-1]:.1f}%")
    print(f"Test Adv Accuracy: {results.test_acc_adv[-1]:.1f}%")


fig, ax = plt.subplots(nrows=2, sharex=True)

ax[0].set_title("Clean data accuracies")
ax[0].plot(results.train_acc, label='Train Acc')
ax[0].plot(results.test_acc, label='Test Acc')
ax[1].set_title("Adversarial data accuracies")
ax[1].plot(results.train_acc_adv, label='Train Acc Adv')
ax[1].plot(results.test_acc_adv, label='Test Acc Adv')
plt.legend()
plt.show()
