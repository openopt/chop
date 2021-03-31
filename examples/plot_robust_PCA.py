"""
Robust PCA
===========

This example fits a Robust PCA model to data.
It uses a hybrid Frank-Wolfe and proximal method.
See description in :func:`chop.optim.minimize_alternating_fw_prox`.


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

for r, p in r_p:
    print(f'r={r} and p={p}')
    U = torch.normal(torch.zeros(1, m, r))
    V = torch.normal(torch.zeros(1, r, n))

    # Low rank component
    L = 10 * utils.bmm(U, V)

    # Sparse component
    S = 100 * torch.normal(torch.zeros(1, m, n))

    S *= (torch.rand_like(S) <= p)

    # Add noise
    N = torch.normal(torch.zeros(1, m, n))

    M = L + S + N
    M = M.to(device)

    @utils.closure
    def sqloss(Z):
        return .5 / M.numel() * torch.linalg.norm((Z - M).squeeze(), ord='fro') ** 2

    rnuc = torch.linalg.norm(L.squeeze(), ord='nuc')
    sL1 = abs(S).sum()

    print(f"Initial L1 norm: {sL1}")
    print(f"Initial Nuclear norm: {rnuc}")

    rank_constraint = chop.constraints.NuclearNormBall(rnuc)
    sparsity_constraint = chop.constraints.L1Ball(sL1)

    lmo = rank_constraint.lmo
    prox = sparsity_constraint.prox

    def things_to_log(kwargs):
        result = (
            torch.linalg.norm(kwargs['y'].squeeze(), ord='nuc').item(),
            abs(kwargs['x']).sum().item(),
            sqloss(kwargs['x'] + kwargs['y'])[0].item()
        )
        return result

    callback = Trace(log_x=False, callable=things_to_log)

    def line_search(kwargs):
        x = kwargs['x']
        y = kwargs['y']
        w = kwargs['w']
        v = kwargs['v']
        q = w + v
        z = x + y
        B = M - z
        A = q - z

        step_size = torch.clamp(utils.bdiv(utils.bdot(A, B), utils.bdot(A, A)), max=1.)
        assert (step_size >= 0).all()
        return step_size

    result = chop.optim.minimize_alternating_fw_prox(sqloss,
                                                     torch.zeros_like(M, device=device),
                                                     torch.zeros_like(M, device=device),
                                                     prox=prox, lmo=lmo,
                                                     L0=1.,
                                                     line_search=line_search,
                                                     max_iter=200,
                                                     callback=callback)

    low_rank_nuc, sparse_comp, f_vals = zip(*callback.trace_callable)

    fig, axes = plt.subplots(3, sharex=True, figsize=(6, 12))
    fig.suptitle(f'r={r} and p={p}')

    axes[0].plot(f_vals)
    axes[0].set_ylim(0, 250)
    axes[0].set_title("Function values")

    axes[1].plot(sparse_comp)
    axes[1].set_title("L1 norm of sparse component")

    axes[2].plot(low_rank_nuc)
    axes[2].set_title("Nuclear Norm of low rank component")

    plt.tight_layout()
    plt.show()
    print(f"L1 norm: {abs(result.x).sum()}")
    print(f"Nuc Norm: {torch.linalg.norm(result.y.squeeze(), ord='nuc')}")
