
"""
Stochastic Robust PCA
===========

This example fits a Robust PCA model to data.
It uses a stochastic hybrid Frank-Wolfe and proximal method.
See description in :func:`chop.stochastic.SplittingProxFW`.


We reproduce the synthetic experimental setting from `[Garber et al. 2018] <https://arxiv.org/pdf/1802.05581.pdf>`_.
We aim to recover :math:`M = L + S + N`, where :math:`L` is rank :math:`p`,
:math:`S` is :math:`p` sparse, and :math:`N` is standard Gaussian elementwise.
"""


import matplotlib.pyplot as plt
import torch
import chop
from chop import utils
from chop.utils.logging import Trace

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = 1000
n = 1000

r_p = [(5, 1e-3),
    #    (5, 3e-3), (25, 1e-3), (25, 3e-3),
    #    (25, 3e-2), (130, 1e-2)
       ]

n_epochs = 100

for r, p in r_p:
    print(f'r={r} and p={p}')
    U = torch.normal(torch.zeros(m, r))
    V = torch.normal(torch.zeros(r, n))

    # Low rank component
    L = 10 * utils.bmm(U, V)

    # Sparse component
    S = 100 * torch.normal(torch.zeros(m, n))

    S *= (torch.rand_like(S) <= p)

    # Add noise
    N = torch.normal(torch.zeros(m, n))

    M = L + S + N
    M = M.to(device)

    def sqloss(Z, M):
        return .5 / Z.numel() * torch.linalg.norm((Z - M).squeeze(), ord='fro') ** 2

    rnuc = torch.linalg.norm(L.squeeze(), ord='nuc')
    sL1 = abs(S).sum()

    print(f"Initial L1 norm: {sL1}")
    print(f"Initial Nuclear norm: {rnuc}")

    rank_constraint = chop.constraints.NuclearNormBall(rnuc)
    sparsity_constraint = chop.constraints.L1Ball(sL1)

    lmo = rank_constraint.lmo
    prox = sparsity_constraint.prox

    Z = torch.zeros_like(M, device=device)
    Z.requires_grad_(True)

    batch_sizes = [100, 250, 500, 1000]
    fig, axes = plt.subplots(ncols=len(batch_sizes), figsize=(18, 10))

    for batch_size, ax in zip(batch_sizes, axes):
        print(f"Batch size: {batch_size}")
        sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(M.size(0))),
                                                batch_size=batch_size,
                                                drop_last=True)

        optimizer = chop.stochastic.SplittingProxFW([Z], lmo=[lmo], prox=[prox],
                                                    lr_lmo='sublinear',
                                                    lr_prox='sublinear',
                                                    normalization='none',
                                                    # weight_decay=1e-8,
                                                    momentum=.9)

        train_losses = []
        losses = []
        sgrad_avg = 0
        n_it = 0
        for it in range(n_epochs):
            for idx in sampler:
                n_it += 1
                optimizer.zero_grad()
                loss = sqloss(Z[idx], M[idx])
                train_losses.append(loss.item())
                loss.backward()
                sgrad = Z.grad.detach().clone()
                sgrad_avg += sgrad
                # for logging
                with torch.no_grad():
                    full_loss = sqloss(Z, M)
                    losses.append(full_loss.item())
                optimizer.step()
        ax.set_title(f"b={batch_size}")
        ax.plot(train_losses, label='training_losses')
        ax.plot(losses, label='loss')
        ax.set_ylim(0, 250)
        ax.legend()
    plt.savefig("robustPCA_stoch.png")
    print("Done.")
