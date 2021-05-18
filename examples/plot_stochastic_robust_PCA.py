
"""
Stochastic Robust PCA
===========

This example fits a Robust PCA model to data.
It uses a stochastic hybrid Frank-Wolfe and proximal method.
See description in :func:`chop.stochastic.SplittingProxFW`.


We reproduce the synthetic experimental setting from `[Garber et al. 2018] <https://arxiv.org/pdf/1802.05581.pdf>`_.
We aim to recover :math:`M = L + S + N`, where :math:`L` is rank :math:`r`,
:math:`S` is :math:`p` sparse, and :math:`N` is standard Gaussian elementwise.
"""


import matplotlib.pyplot as plt
import torch
import chop
from chop import utils
from time import time
import numpy as np

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = 1000
n = 1000

r_p = [(5, 1e-3),
       (5, 3e-3), (25, 1e-3), (25, 3e-3),
       (25, 3e-2), (130, 1e-2)
       ]

n_epochs = 1000

sqloss = torch.nn.MSELoss(reduction='sum')
lam = 1. / np.sqrt(m)
freq = 100

for r, p in r_p:
    print(f'r={r} and p={p}')
    U = torch.normal(torch.zeros(m, r))
    V = torch.normal(torch.zeros(r, n))

    # Low rank component
    L = 10 * utils.bmm(U, V).to(device)

    # Sparse component
    S = 100 * torch.normal(torch.zeros(m, n)).to(device)

    S *= (torch.rand_like(S) <= p)

    # Add noise
    N = torch.normal(torch.zeros(m, n)).to(device)

    M = L + S + N
    M = M.to(device)
    
    # From Candès paper
    mu = (m * n) / (8 * torch.linalg.norm(M.view(-1), ord=1))

    rnuc = torch.linalg.norm(L.squeeze(), ord='nuc')
    sL1 = abs(S).sum()

    print(f"Initial L1 norm: {sL1}")
    print(f"Initial Nuclear norm: {rnuc}")

    rank_constraint = chop.constraints.NuclearNormBall(rnuc)
    rank_penalty = chop.penalties.NuclearNorm(1.)
    sparsity_constraint = chop.constraints.L1Ball(sL1)
    sparsity_penalty = chop.penalties.L1(lam)

    lmo = rank_penalty.lmo
    # lmo = rank_constraint.lmo
    # prox = sparsity_constraint.prox
    prox = sparsity_penalty.prox
    prox_lr = rank_penalty.prox

    # batch_sizes = [100, 250, 500, 1000]
    batch_sizes = [1000]
    fig, axes = plt.subplots(nrows=2, ncols=max(len(batch_sizes), 2), figsize=(18, 10), sharey=True)
    fig.suptitle(f'r={r} and p={p}')

    for batch_size, ax_it, ax_time in zip(batch_sizes, axes[0], axes[1]):
        Z = torch.zeros_like(M, device=device)
        Z.requires_grad_(True)
        print(f"Batch size: {batch_size}")
        dataset = torch.utils.data.TensorDataset(Z, M)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        momentum = .9 if batch_size != m else 0.

        optimizer = chop.stochastic.SplittingProxFW([Z], lmo=[lmo],
                                                    prox1=[prox],
                                                    prox2=[prox_lr],
                                                    lr='sublinear',
                                                    lipschitz=1.,
                                                    normalization='none',
                                                    momentum=momentum,
                                                    generalized_lmo=True)


        train_losses = []
        times = []
        losses = []
        sgrad_avg = 0
        n_it = 0
        start = time()
        for it in range(n_epochs):
            for zi, mi in loader:
                n_it += 1
                optimizer.zero_grad()
                loss = mu * sqloss(zi, mi) * Z.size(0) / zi.size(0)
                loss.backward()
                sgrad = Z.grad.detach().clone()
                sgrad_avg += sgrad
                # for logging
                if n_it % freq == 0:
                    with torch.no_grad():
                        times.append(time() - start)
                        full_loss = mu * sqloss(Z, M) + torch.linalg.norm(optimizer.state[Z]['y'], ord='nuc') + lam * optimizer.state[Z]['x'].abs().sum()
                        print(full_loss / torch.linalg.norm(M))
                        train_losses.append(loss.item())
                        losses.append(full_loss.item())
                optimizer.step()
        donetime = time()

        # Get sparse and LR component
        state = optimizer.state[Z]
        sparse_comp = state['x']
        lr_comp = state['y']

        # Plots
        ax_it.set_title(f"b={batch_size}")
        ax_it.plot(train_losses, label='mini-batch loss')
        ax_it.plot(losses, label='full loss')
        ax_it.set_xlabel('iterations')
        ax_time.plot(times, train_losses, label='minibatch loss')
        ax_time.plot(times, losses, label='full loss')
        ax_time.set_xlabel('time (s)')
        ax_it.set_yscale('log')
        ax_it.legend()
        print(f"Low rank loss: {torch.linalg.norm(L - lr_comp) / torch.linalg.norm(L)}")
        print(f"Sparse loss: {torch.linalg.norm(S - sparse_comp) / torch.linalg.norm(S)}")
        print(f"Reconstruction loss: {torch.linalg.norm(M - sparse_comp - lr_comp) / torch.linalg.norm(M)}")
        print(f"Time: {times[-1]}s")
        break
    fig.show()
    fig.savefig(f"robustPCA_{r, p}.png")
print("Done.")
