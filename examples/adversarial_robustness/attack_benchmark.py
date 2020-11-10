from functools import partial

import torch
from tqdm import tqdm

import constopt as cpt
from constopt.data import load_cifar10

from robustbench.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 200
n_examples = 10000
loader = load_cifar10(batch_size=batch_size, data_dir='~/datasets')

model_name = 'Standard'
model = load_model(model_name, norm='Linf').to(device)
criterion = torch.nn.CrossEntropyLoss()

# Define the perturbation constraint set
alpha = 8 / 255.
constraint = cpt.constraints.LinfBall(alpha)


def image_constraint_prox(delta, step_size=None, data=None):
    """Projects perturbation delta
    so that 0. <= data + delta <= 1."""

    adv_img = torch.clamp(data + delta, 0, 1)
    delta = adv_img - data
    return delta


def loss_fun(delta, data, target):
    adv_input = data + delta
    return -criterion(model(adv_input), target)


def prox(delta, step_size=None, data=None, target=None):
    delta = constraint.prox(delta, step_size)
    delta = image_constraint_prox(delta, step_size, data)

    return delta


print(f"Evaluating model {model_name} on L{constraint.p} ball({alpha}).")

n_correct = 0
n_correct_adv = 0

for k, (data, target) in tqdm(enumerate(loader), total=len(loader)):
    data = data.to(device)
    target = target.to(device)

    loss_f = partial(loss_fun, data=data, target=target)
    prox_f = partial(prox, data=data, target=target)

    def f_and_grad(delta):
        loss = loss_f(delta)
        loss.backward()
        return loss, delta.grad

    delta0 = torch.zeros_like(data, dtype=data.dtype)

    sol = cpt.optim.minimize_pgd_madry(delta0, f_and_grad,
                                       prox_f,
                                       constraint.lmo,
                                       step_size=alpha / 20.,
                                       max_iter=20, callback=None)

    label = torch.argmax(model(data), dim=-1)
    delta = sol
    adv_label = torch.argmax(model(data + delta), dim=-1)

    n_correct += (label == target).sum().item()
    n_correct_adv += (adv_label == target).sum().item()

    accuracy = n_correct / ((k + 1) * batch_size)
    accuracy_adv = n_correct_adv / ((k + 1) * batch_size)

accuracy = n_correct / n_examples
accuracy_adv = n_correct_adv / n_examples

print(f"Accuracy: {accuracy:.4f}")
print(f"RobustAccuracy: {accuracy_adv:.4f}")
