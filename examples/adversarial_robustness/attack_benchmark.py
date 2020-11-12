from constopt.optim import minimize_pgd, minimize_pgd_madry
from constopt.utils import closure

import torch
from tqdm import tqdm

import constopt as cpt
from constopt.data import load_cifar10
from constopt.adversary import Adversary

from robustbench.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 100
n_examples = 10000
loader = load_cifar10(batch_size=batch_size, data_dir='~/datasets')

model_name = 'Standard'
model = load_model(model_name, norm='Linf').to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Define the perturbation constraint set
max_iter = 20
alpha = 8 / 255.
constraint = cpt.constraints.LinfBall(alpha)


print(f"Evaluating model {model_name} on L{constraint.p} ball({alpha}).")

n_correct = 0
n_correct_adv_pgd_madry = 0
n_correct_adv_pgd = 0

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

    adversary_pgd = Adversary(minimize_pgd)
    adversary_pgd_madry = Adversary(minimize_pgd_madry)

    _, delta_pgd = adversary_pgd.perturb(data, target, model, criterion,
                                         use_best=False,
                                         prox=prox,
                                         max_iter=max_iter)

    _, delta_pgd_madry = adversary_pgd_madry.perturb(data, target, model,
                                                     criterion,
                                                     use_best=False,
                                                     prox=prox,
                                                     lmo=constraint.lmo,
                                                     step=2 * constraint.alpha / max_iter, 
                                                     max_iter=max_iter)

    label = torch.argmax(model(data), dim=-1)
    n_correct += (label == target).sum().item()

    adv_label_pgd_madry = torch.argmax(model(data + delta_pgd_madry), dim=-1)
    n_correct_adv_pgd_madry += (adv_label_pgd_madry == target).sum().item()

    adv_label_pgd = torch.argmax(model(data + delta_pgd), dim=-1)
    n_correct_adv_pgd += (adv_label_pgd == target).sum().item()


accuracy = n_correct / n_examples
accuracy_adv_pgd_madry = n_correct_adv_pgd_madry / n_examples
accuracy_adv_pgd = n_correct_adv_pgd / n_examples

print(f"Accuracy: {accuracy:.4f}")
print(f"RobustAccuracy: {accuracy_adv_pgd_madry:.4f}")
print(f"RobustAccuracy: {accuracy_adv_pgd:.4f}")
