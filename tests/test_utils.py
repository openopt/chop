"""Tests for utility functions"""

from chop.utils import closure
import torch
from torch import nn
from chop import utils


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Set up random regression problem
alpha = 1.
n_samples, n_features = 20, 15
X = torch.rand((n_samples, n_features))
w = torch.rand(n_features)
w = alpha * w / sum(abs(w))
y = X.mv(w)
# Logistic regression: \|y\|_\infty <= 1
y = abs(y / y.max())

tol = 4e-3

batch_size = 20
d1 = 10
d2 = 5
x0 = torch.ones(batch_size, d1, d2)


def test_jacobian_batch():
    def loss(x):
        return (x.view(x.size(0), -1) ** 2).sum(-1)

    val, jac = utils.get_func_and_jac(loss, x0)

    assert jac.eq(2 * x0).all()


def test_jacobian_single_sample():
    def loss(x):
        return (x ** 2).sum()

    x0 = torch.rand(1, d1, d2)
    val, jac = utils.get_func_and_jac(loss, x0)

def test_closure():

    @utils.closure
    def loss(x):
        return (x.view(x.size(0), -1) ** 2).sum(-1)

    val, grad = loss(x0)
    assert val.eq(torch.ones(batch_size) * (d1 * d2)).all()
    assert grad.eq(2 * x0).all()


def test_init_lipschitz():
    criterion = nn.MSELoss(reduction='none')

    @closure
    def loss_fun(X):
        return criterion(X.mv(w), y)

    L = utils.init_lipschitz(loss_fun, X.detach().clone().requires_grad_(True))
    print(L)


def test_bmm():
    """
    Check shape returned by batch matmul
    """
    for _ in range(10):
        t1 = torch.rand(4, 3, 32, 35)
        t2 = torch.rand(4, 3, 35, 32)

        res = utils.bmm(t1, t2)
        assert res.shape == (4, 3, 32, 32)


def test_bmv():
    """
    Check shape returns of batch mat-vec multiply
    """
    for _ in range(10):
        mat = torch.rand(4, 3, 32, 35)
        vec = torch.rand(4, 3, 35)

        res = utils.bmv(mat, vec)
        assert res.shape == (4, 3, 32)


def test_power_iteration():
    """
    Checks our power iteration method against torch.svd
    """
    mat = torch.rand(4, 3, 32, 35)
    mat.to(device)
    # Ground truth
    U, S, V = torch.svd(mat)
    u, s, v = utils.power_iteration(mat, n_iter=10)

    # First singular value should be the same
    assert torch.allclose(S[..., 0], s, atol=1e-5), (S[..., 0] - s)

    outer = U[..., 0].unsqueeze(-1) * V[..., 0].unsqueeze(-2)
    outer_pi = u.unsqueeze(-1) * v.unsqueeze(-2)

    # Rank 1 approx should be the same
    assert torch.allclose(outer, outer_pi), (outer - outer_pi)
