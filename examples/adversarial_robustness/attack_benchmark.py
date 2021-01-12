"""
Benchmark of attacks.
========================
"""
import torch
from tqdm import tqdm

import chop
from chop.optim import minimize_frank_wolfe, minimize_pgd, minimize_pgd_madry, minimize_three_split
from chop.data import load_cifar10
from chop.adversary import Adversary

from robustbench.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 50
n_examples = 10000
loaders = load_cifar10(test_batch_size=batch_size, data_dir='~/datasets')
loader = loaders.test

model_name = 'Engstrom2019Robustness'
model = load_model(model_name, norm='Linf').to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Define the perturbation constraint set
max_iter = 20
alpha = 8 / 255.
constraint = chop.constraints.LinfBall(alpha)


print(f"Evaluating model {model_name} on L{constraint.p} ball({alpha}).")

n_correct = 0
n_correct_adv_pgd_madry = 0
n_correct_adv_pgd = 0
n_correct_adv_split = 0
n_correct_adv_fw = 0

adversary_pgd = Adversary(minimize_pgd)
adversary_pgd_madry = Adversary(minimize_pgd_madry)
adversary_split = Adversary(minimize_three_split)
adversary_fw = Adversary(minimize_frank_wolfe)

for k, (data, target) in tqdm(enumerate(loader), total=len(loader)):
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


    _, delta_pgd = adversary_pgd.perturb(data, target, model, criterion,
                                         use_best=True,
                                         step='backtracking',
                                         prox=prox,
                                         max_iter=max_iter)

    delta_pgd_madry = torch.zeros_like(data)
    # _, delta_pgd_madry = adversary_pgd_madry.perturb(data, target, model,
    #                                                  criterion,
    #                                                  use_best=False,
    #                                                  prox=prox,
    #                                                  lmo=constraint.lmo,
    #                                                  step=2. / max_iter,
    #                                                  max_iter=max_iter)

    delta_split = torch.zeros_like(data)
    # _, delta_split = adversary_split.perturb(data, target, model,
    #                                          criterion,
    #                                          use_best=False,
    #                                          prox1=constraint.prox,
    #                                          prox2=image_constraint_prox,
    #                                          max_iter=max_iter)

    delta_fw = torch.zeros_like(data)
    # _, delta_fw = adversary_fw.perturb(data, target, model, criterion,
    #                                    lmo=constraint.lmo,
    #                                    step='sublinear',
    #                                    max_iter=max_iter
    #                                    )

    label = torch.argmax(model(data), dim=-1)
    n_correct += (label == target).sum().item()

    adv_label_pgd_madry = torch.argmax(model(data + delta_pgd_madry), dim=-1)
    n_correct_adv_pgd_madry += (adv_label_pgd_madry == target).sum().item()

    adv_label_pgd = torch.argmax(model(data + delta_pgd), dim=-1)
    n_correct_adv_pgd += (adv_label_pgd == target).sum().item()

    adv_label_split = torch.argmax(model(data + delta_split), dim=-1)
    n_correct_adv_split += (adv_label_split == target).sum().item()

    adv_label_fw = torch.argmax(model(data + delta_fw), dim=-1)
    n_correct_adv_fw += (adv_label_fw == target).sum().item()


accuracy = n_correct / n_examples
accuracy_adv_pgd_madry = n_correct_adv_pgd_madry / n_examples
accuracy_adv_pgd = n_correct_adv_pgd / n_examples
accuracy_adv_split = n_correct_adv_split / n_examples
accuracy_adv_fw = n_correct_adv_fw / n_examples

print(f"Accuracy: {accuracy:.4f}")
print(f"RobustAccuracy PGD Madry: {accuracy_adv_pgd_madry:.4f}")
print(f"RobustAccuracy PGD: {accuracy_adv_pgd:.4f}")
print(f"RobustAccuracy Splitting: {accuracy_adv_split:.4f}")
print(f"RobustAccuracy FW: {accuracy_adv_fw:.4f}")
