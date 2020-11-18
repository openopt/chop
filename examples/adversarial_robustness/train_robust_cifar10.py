from chop.adversary import Adversary
import torch
from tqdm import tqdm

import chop

from torch.optim import SGD

from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

n_epochs = 128
batch_size = 100

trainloader = chop.data.load_cifar10(batch_size=batch_size,
                                     train=True,
                                     data_dir='~/datasets')
testloader = chop.data.load_cifar10(batch_size=batch_size,
                                    data_dir='~/datasets')

model = models.resnet18(pretrained=False)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)

# Define the perturbation constraint set
max_iter_train = 7
max_iter_test = 20
alpha = 8. / 255
constraint = chop.constraints.LinfBall(alpha)
criterion_adv = torch.nn.CrossEntropyLoss(reduction='none')

print(f"Training on L{constraint.p} ball({alpha}).")


adversary = Adversary(chop.optim.minimize_pgd_madry)

for _ in range(n_epochs):

    # Train
    n_correct = 0
    n_correct_adv = 0

    model.train()

    for k, (data, target) in tqdm(enumerate(trainloader), total=len(trainloader)):
        data = data.to(device)
        target = target.to(device)

        def image_constraint_prox(delta, step_size=None):
            """Projects perturbation delta
            so that 0. <= data + delta <= 1."""

            adv_img = torch.clamp(data + delta, 0, 1)
            delta = adv_img - data
            return delta

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

        output = model(data)
        output_adv = model(data + delta)
        loss = criterion(output_adv, target)
        loss.backward()

        pred = torch.argmax(output, dim=-1)
        pred_adv = torch.argmax(output_adv, dim=-1)

        n_correct += (pred == target).sum().item()
        n_correct_adv += (pred_adv == target).sum().item()

    print(f"Train Accuracy: {n_correct / 50000.}")
    print(f"Train Adv Accuracy: {n_correct_adv / 50000.}")

    # Test
    n_correct = 0
    n_correct_adv = 0

    model.eval()

    for k, (data, target) in tqdm(enumerate(testloader), total=len(testloader)):
        data = data.to(device)
        target = target.to(device)

        def image_constraint_prox(delta, step_size=None):
            """Projects perturbation delta
            so that 0. <= data + delta <= 1."""

            adv_img = torch.clamp(data + delta, 0, 1)
            delta = adv_img - data
            return delta

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

    print(f"Test Accuracy: {n_correct / 10000.}")
    print(f"Test Adv Accuracy: {n_correct_adv / 10000.}")