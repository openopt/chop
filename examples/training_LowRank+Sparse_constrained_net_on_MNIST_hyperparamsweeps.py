"""
Constrained Neural Network Training.
======================================
Trains a ResNet model on CIFAR10 using constraints on the weights.
This example is inspired by the official PyTorch MNIST example, which
can be found [here](https://github.com/pytorch/examples/blob/master/mnist/main.py).
"""
from __future__ import print_function
import torch.nn.functional as F
import torch.nn as nn
import argparse

import numpy as np


import torch
from torch import nn
from torch.nn import functional as F

import chop

import wandb


# Hyperparam setup
default_config = {
    'lr': 1e-4,
    'batch_size': 64,
    'momentum': .5,
    'weight_decay': 1e-5,
    'lr_bias': 0.005,
    'grad_norm': 'none',
    'l1_constraint_size': 30,
    'nuc_constraint_size': 1e2,
    'epochs': 2,
    'seed': 1
}

wandb.init(project='low-rank_sparse_mnist', config=default_config)
config = wandb.config


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_sparsity_and_rank(opt):
    nnzero = 0
    n_params = 0
    total_rank = 0
    max_rank = 0

    for group in opt.param_groups:
        for p in group['params']:
            state = opt.state[p]
            nnzero += (state['x'] !=0 ).sum()
            n_params += p.numel()
            ranks = torch.linalg.matrix_rank(state['y'])
            total_rank += ranks.sum()
            max_rank += min(p.shape) * ranks.numel()

    return nnzero / n_params, total_rank / max_rank


def train(args, model, device, train_loader, opt, opt_bias, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        opt_bias.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if loss.isnan():
            break
        opt.step()
        opt_bias.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"Train Loss": loss.item()})
    sparsity, rank = get_sparsity_and_rank(opt)
    wandb.log({"Sparsity": sparsity,
               "Rank": rank})


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})


def main():

    wandb.init()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr',  default=1e-2, metavar='LR',
                        help='learning rate (default: "sublinear")')
    parser.add_argument('--lr_bias', default=0.005, type=float, metavar='LR_BIAS',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='Optimizer momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='W',
                        help='Optimizer weight decay (default: 0.)')
    parser.add_argument('--grad_norm', type=str, default='gradient',
                        help='Gradient normalization options')
    parser.add_argument('--nuc_constraint_size', type=float, default=1e2,
                        help='Size of the Nuclear norm Ball constraint')
    parser.add_argument('--l1_constraint_size', type=float, default=30,
                        help='Size of the ell-1 norm Ball constraint')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.lr != 'sublinear':
        args.lr = float(args.lr)

    wandb.config.update(args, allow_val_change=True)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = chop.utils.data.MNIST("~/datasets/")
    loaders = dataset.loaders(args.batch_size, args.test_batch_size)

    model = Net().to(device)
    constraints_sparsity = chop.constraints.make_model_constraints(model,
                                                                   ord=1,
                                                                   value=args.l1_constraint_size,
                                                                   constrain_bias=False)
    constraints_low_rank = chop.constraints.make_model_constraints(model,
                                                                   ord='nuc',
                                                                   value=args.nuc_constraint_size,
                                                                   constrain_bias=False)
    proxes = [constraint.prox if constraint else None
              for constraint in constraints_sparsity]
    lmos = [constraint.lmo if constraint else None
            for constraint in constraints_low_rank]

    proxes_lr = [constraint.prox if constraint else None
                 for constraint in constraints_low_rank]

    chop.constraints.make_feasible(model, proxes)
    chop.constraints.make_feasible(model, proxes_lr)

    optimizer = chop.stochastic.SplittingProxFW(model.parameters(), lmos,
                                                proxes,
                                                lr_lmo=args.lr,
                                                lr_prox=args.lr,
                                                momentum=args.momentum,
                                                weight_decay=args.weight_decay,
                                                normalization=args.grad_norm)

    bias_params = (param for name, param in model.named_parameters()
                   if chop.constraints.is_bias(name, param))
    bias_opt = chop.stochastic.PGD(bias_params, lr=args.lr_bias)


    wandb.watch(model, log_freq=1, log='all')

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, loaders.train, optimizer, bias_opt, epoch)
        test(args, model, device, loaders.test)


if __name__ == '__main__':
    main()
