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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

m = 1000
n = 1000
r = 5
p = 0.001

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
    return .5 * torch.linalg.norm((Z - M).squeeze(), ord='fro') ** 2

rnuc = torch.linalg.norm(L.squeeze(), ord='nuc')
sL1 = abs(S).sum()

print(f"Initial sparsity: {sL1}")
print(f"Initial nuclear norm: {rnuc}")

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
    assert (step_size >= 0).all(), 'WTF'
    return step_size


result = chop.optim.minimize_alternating_fw_prox(sqloss, torch.zeros_like(M, device=device), torch.zeros_like(M, device=device),
                                                 prox=prox, lmo=lmo,
                                                 L0=1.,
                                                 line_search=line_search,
                                                 max_iter=200,
                                                 callback=callback)


low_rank_nuc, sparse_comp, f_vals = zip(*callback.trace_callable)

fig, axes = plt.subplots(3, sharex=True, figsize=(12, 4))
axes[0].plot(f_vals)
axes[0].set_title("Function values")

axes[1].plot(sparse_comp)
axes[1].set_title("L1 norm of sparse component")

axes[2].plot(low_rank_nuc)
axes[2].set_title("Nuclear Norm of low rank component")


plt.tight_layout()
plt.show()

print(f"Sparsity: {abs(result.x).sum()}")
print(f"Nuc Norm: {torch.linalg.norm(result.y.squeeze(), ord='nuc')}")
