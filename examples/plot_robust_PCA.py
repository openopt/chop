"""
Robust PCA
===========

This example fits a Robust PCA model to data.
It uses a hybrid Frank-Wolfe and proximal method.
See description in optim.minimize_alternating_fw_prox.


We reproduce the synthetic experimental setting from [Garber et al. 2018].
"""


import matplotlib.pyplot as plt
import torch
import chop
from chop import utils
from chop.utils.logging import Trace

m = 1000
n = 1000
r = 5
p = 0.001

U = torch.normal(torch.zeros(1, m, r))
V = torch.normal(torch.zeros(1, r, n))

# low rank part
L = 10 * utils.bmm(U, V)

S = 10 * torch.normal(torch.zeros(1, m, n))

S *= torch.bernoulli(p * torch.ones_like(S))

M = L + S

@utils.closure
def sqloss(Z):
    return .5 * torch.linalg.norm((Z - M).squeeze(), ord='fro') ** 2

rnuc = torch.linalg.norm(L.squeeze(), ord='nuc')
sL1 = abs(S).sum()

print(f"Initial sparsity: {sL1}")
print(f"Initial nuclear norm: {rnuc}")

rank_constraint = chop.constraints.NuclearNormBall(rnuc)
sparsity_constraint = chop.constraints.L1Ball(sL1)


lmo = rank_constraint.lmo
prox = sparsity_constraint.prox

callback = Trace(log_x=False, callable=lambda kwargs: (torch.linalg.norm(kwargs['y'].squeeze(), ord='nuc').item(),
                                                           abs(kwargs['x']).sum().item()))

result = chop.optim.minimize_alternating_fw_prox(sqloss, torch.zeros_like(M), torch.zeros_like(M),
                                                 prox=prox, lmo=lmo,
                                                 L0=1., max_iter=200,
                                                 callback=callback)


f_vals = callback.trace_f
low_rank_nuc, sparse_comp = zip(*callback.trace_callable)

fig, axes = plt.subplots(3)
axes[0].plot(callback.trace_f)
axes[0].set_title("Function values")

axes[1].plot(sparse_comp)
axes[1].set_title("L1 norm of sparse component")

axes[2].plot(low_rank_nuc)
axes[2].set_title("Nuclear Norm of low rank component")
# plt.yscale('log')
plt.tight_layout()
plt.show()
plt.savefig("robustPCA.png")

print(f"Sparsity: {abs(result.y).sum()}")
print(f"Nuc Norm: {torch.linalg.norm(result.x.squeeze())}")