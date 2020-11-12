from constopt.utils import closure

import torch
from tqdm import tqdm

import constopt as cpt
from constopt.data import load_cifar10

from robustbench.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 20
n_examples = 10000
loader = load_cifar10(batch_size=batch_size, data_dir='~/datasets')

model_name = 'Standard'
model = load_model(model_name, norm='Linf').to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Define the perturbation constraint set
n_iter = 20
alpha = 8 / 255.
constraint = cpt.constraints.LinfBall(alpha)


print(f"Evaluating model {model_name} on L{constraint.p} ball({alpha}).")

n_correct = 0
n_correct_adv_pgd_madry = 0
n_correct_adv_split = 0

for k, (data, target) in tqdm(enumerate(loader), total=len(loader)):
    data = data.to(device)
    target = target.to(device)

    @closure
    def loss_fun(delta):
        adv_input = data + delta
        return -criterion(model(adv_input), target)

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

    delta0 = torch.zeros_like(data, dtype=data.dtype)

    print("PGD Madry")
    delta_pgd_madry = cpt.stochastic.minimize_pgd_madry(delta0, loss_fun,
                                                   prox,
                                                   constraint.lmo,
                                                   step_size=2. / n_iter,
                                                   max_iter=n_iter,
                                                   callback=None).x

    print("Splitting.")
    delta_split = cpt.stochastic.minimize_three_split(delta0, loss_fun,
                                                 prox1=constraint.prox,
                                                 prox2=image_constraint_prox,
                                                 step_size=None,
                                                 max_iter=n_iter,
                                                 callback=None
                                                 ).x

    label = torch.argmax(model(data), dim=-1)
    n_correct += (label == target).sum().item()

    adv_label_pgd_madry = torch.argmax(model(data + delta_pgd_madry), dim=-1)
    n_correct_adv_pgd_madry += (adv_label_pgd_madry == target).sum().item()

    adv_label_split = torch.argmax(model(data + delta_split), dim=-1)
    n_correct_adv_split += (adv_label_split == target).sum().item()


accuracy = n_correct / n_examples
accuracy_adv_pgd_madry = n_correct_adv_pgd_madry / n_examples
accuracy_adv_split = n_correct_adv_split / n_examples

print(f"Accuracy: {accuracy:.4f}")
print(f"RobustAccuracy: {accuracy_adv_pgd_madry:.4f}")
print(f"RobustAccuracy: {accuracy_adv_split:.4f}")
