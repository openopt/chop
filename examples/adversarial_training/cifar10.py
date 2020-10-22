import os
from argparse import ArgumentParser

from easydict import EasyDict

from tqdm import tqdm

import numpy as np
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet18

import constopt
from constopt.adversary import Adversary
from constopt.optim import PGD, PGDMadry, FrankWolfe, MomentumFrankWolfe
from constopt.data_utils import ld_cifar10


parser = ArgumentParser()
parser.add_argument("--inner-alg", type=str, default="pgd-madry", help="pgd | pgd-madry | fw | mfw | none")
parser.add_argument("--inner-iter", type=int, default=7)
parser.add_argument("--inner-iter-test", type=int, default=20)
parser.add_argument("--eps", type=float, default=8./255)
# parser.add_argument("--eps-test", type=float, default=None)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--inner-step-size", type=float)
parser.add_argument("--random_init", action="store_true")
parser.add_argument("--nb-epochs", default=50)
parser.add_argument("--p", default='inf', help="2 | inf")


def main(args):
    # Setup
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data Loaders
    loader = ld_cifar10()
    train_loader = loader.train
    test_loader = loader.test

    # Model setup
    if args.model == "resnet18":
        model = resnet18()
    else:
        raise ValueError("Must use resnet18 for now.")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # TODO use SOTA schedulers etc...
    # Outer optimization parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Inner optimization parameters
    p = np.inf if args.p == "inf" else int(args.p)
    constraint = constopt.constraints.make_LpBall(alpha=args.eps, p=p)  # also used for testing

    random_init = args.random_init  # Sample the starting optimization point uniformly at random in the constraint set

    if args.inner_alg == "pgd":
        adv_opt_class = PGD
        if not args.inner_step_size:
            inner_step_size = 5e4 * 2.5 / args.inner_iter
    elif args.inner_alg == "pgd-madry":
        adv_opt_class = PGDMadry  # To beat
        if not args.inner_step_size:
            inner_step_size = 2.5 / args.inner_iter
    elif args.inner_alg == "fw":
        adv_opt_class = FrankWolfe
        if not args.inner_step_size:
            inner_step_size = None
    elif args.inner_alg == "mfw":
        adv_opt_class = MomentumFrankWolfe
        if not args.inner_step_size:
            inner_step_size = None
    elif args.inner_alg == "none":
        adv_opt_class = None
        inner_step_size = None
    else:
        raise ValueError("This algorithm isn't implemented yet.")

    # Use default values for now
    step_size_test = {
        PGD.name: 5e4 * 2.5 * constraint.alpha / args.inner_iter_test,
        PGDMadry.name: 2.5 * constraint.alpha / args.inner_iter_test,
        FrankWolfe.name: None,
        MomentumFrankWolfe.name: None
    }


    # Logging
    writer = SummaryWriter(os.path.join("logging/cifar10/",
                                        adv_opt_class.name if adv_opt_class else "Clean"))

    # Training loop
    for epoch in range(args.nb_epochs):
        model.train()
        train_loss = 0.
        for data, target in tqdm(train_loader, desc=f'Training epoch {epoch}/{args.nb_epochs - 1}'):
            data, target = data.to(device), target.to(device)

            adv = Adversary(data.shape, constraint, adv_opt_class,
                            device=device, random_init=random_init)
            _, delta = adv.perturb(data, target, model, criterion,
                                   inner_step_size,
                                   iterations=args.inner_iter,
                                   use_best=True,
                                   tol=1e-7)
            optimizer.zero_grad()
            adv_loss = criterion(model(data + delta), target)
            adv_loss.backward()
            optimizer.step()

            train_loss += adv_loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        print(f'Training loss: {train_loss:.3f}')
        # TODO: get accuracy

        # Evaluate on clean and adversarial test data

        model.eval()
        report = EasyDict(acc_cln=0., acc_pgd=0., acc_pgd_madry=0, acc_fw=0., acc_mfw=0.) 
        nb_test = 0
        for data, target in tqdm(test_loader, desc=f'Val epoch {epoch}/{args.nb_epochs - 1}'):
            data, target = data.to(device), target.to(device)
            
            def get_nb_correct(alg_class):
                adv = Adversary(data.shape,
                                constraint,
                                alg_class,
                                device=device,
                                random_init=args.random_init)
                if alg_class:
                    _, delta = adv.perturb(data, target, model, criterion,
                                           step_size_test[alg_class.name],
                                           iterations=args.inner_iter_test,
                                           use_best=True,
                                           tol=1e-7)
                else:
                    delta = 0.

                _, pred = model(data + delta).max(1)
                return pred.eq(target).sum().item()

            nbs_correct = ((key, get_nb_correct(alg_class))
                           for key, alg_class in (("cln", None),
                                                  ("pgd", PGD),
                                                  ("pgd_madry", PGDMadry),
                                                  ("fw", FrankWolfe),
                                                  ("mfw", MomentumFrankWolfe))
                           )

            for key, nb_correct in nbs_correct:
                report["acc_" + key] += nb_correct

            nb_test += data.size(0)

        for key, val in report.items():
            if "acc" in key:
                report[key] /= nb_test

        descs = ("clean examples", "adversarial examples PGD", "adversarial examples PGDMadry",
                 "adversarial examples FW", "adversarial examples MFW")

        for acc_val, desc in zip((report.acc_cln,
                                  report.acc_pgd, report.acc_pgd_madry,
                                  report.acc_fw, report.acc_mfw),
                                 descs):
            print(f'Val acc on {desc} (%): {acc_val * 100.:.3f}')

        writer.add_scalars("Test/Acc", report, epoch)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
