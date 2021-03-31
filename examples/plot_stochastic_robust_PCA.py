
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
        return .5 / M.numel() * torch.linalg.norm((Z - M).squeeze(), ord='fro') ** 2

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

    sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(M.size(0))),
                                            batch_size=100,
                                            drop_last=False)

    optimizer = chop.stochastic.SplittingProxFW([Z], lmo=[lmo], prox=[prox],
                                                lr_lmo='sublinear',
                                                lr_prox='sublinear',
                                                normalization='none')

    train_losses = []
    losses = []

    for it in range(n_epochs):
        for idx in sampler:
            optimizer.zero_grad()
            loss = sqloss(Z[idx], M[idx])
            # for logging
            with torch.no_grad():
                full_loss = sqloss(Z, M)
                losses.append(full_loss.item())
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()


    plt.plot(train_losses, label='training_losses')
    plt.plot(losses, label='loss')
    plt.ylim(0, 250)
    plt.legend()
    print("Done.")
