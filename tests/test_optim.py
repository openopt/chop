"""Tests for constrained optimizers"""

import torch
from torch.autograd import Variable
import pytest
import constopt

torch.manual_seed(0)

# Set up random regression problem
alpha = 1.
n_samples, n_features = 20, 15
X = torch.rand((n_samples, n_features))
w = torch.rand(n_features) 
w = alpha * w / sum(abs(w))
y = X.mv(w)
# Logistic regression: \|y\|_\infty <= 1
y = abs(y / y.max())

tol = 1e-3

def test_FW():
    constraint = constopt.constraints.L1Ball(alpha)
    w_t = Variable(torch.zeros_like(w), requires_grad=True)
    optimizer = constopt.optim.FrankWolfe([w_t], constraint.lmo)
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(X.mv(w_t), y)
    ii = 0
    while loss.item() > tol:
        if ii == 1000:
            raise ValueError("Algo should have converged.")
        print(loss.item())
        loss.backward()
        optimizer.step()
        loss = criterion(X.mv(w_t), y)

        ii += 1

        
