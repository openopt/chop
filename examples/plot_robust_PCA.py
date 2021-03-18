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

rank_constraint = chop.constraints.NuclearNormBall(torch.linalg.norm(L.squeeze(), ord='nuc'))
sparsity_constraint = chop.constraints.L1Ball(abs(S).sum())


lmo = rank_constraint.lmo
prox = sparsity_constraint.prox

callback = Trace(log_x=False)

result = chop.optim.minimize_alternating_fw_prox(sqloss, torch.zeros_like(M), torch.zeros_like(M),
                                                 prox=prox, lmo=lmo,
                                                 L0=1., max_iter=200,
                                                 callback=callback)


plt.plot(callback.trace_f)
# plt.yscale('log')
plt.tight_layout()
plt.show()
plt.savefig("robustPCA.png")

print(f"Sparsity: {abs(result.y).sum()}")
print(f"Nuc Norm: {torch.linalg.norm(result.x.squeeze())}")