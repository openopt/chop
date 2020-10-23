import torch

from robustbench.data import load_cifar10
from robustbench.utils import load_model

from constopt.adversary import Adversary
from constopt.optim import PGD, PGDMadry, FrankWolfe, MomentumFrankWolfe
from constopt.constraints import LinfBall

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
data, target = load_cifar10(n_examples=100)
model = load_model(model_name='Carmon2019Unlabeled', norm='Linf')
criterion = torch.nn.CrossEntropyLoss()
eps = 8. / 255
constraint = LinfBall(eps)
n_iter = 20

step_size_test = {
    PGD.name: 5e4 * 2.5 * constraint.alpha / n_iter,
    PGDMadry.name: 2.5 / n_iter,
    FrankWolfe.name: None,
    MomentumFrankWolfe.name: None
}

for alg_class in PGD, PGDMadry, FrankWolfe, MomentumFrankWolfe:

    adv = Adversary(data.shape, constraint, alg_class, device)
    adv_loss, delta = adv.perturb(data, target, model, criterion,
                                  step_size=step_size_test[alg_class.name],
                                  iterations=n_iter)
    _, pred = model(data + delta).max(1)
    accuracy = pred.eq(target).sum().item() / data.size(0)
    print(f"Robust accuracy on {alg_class.name} (%): {accuracy * 100.:.3f}")
