from functools import wraps


def close(f):
    @wraps(f)
    def wrapper(x, return_gradient=True, *args, **kwargs):
        """Adds gradient computation when calling function"""
        val = f(x, *args, **kwargs)
        if not return_gradient:
            return val
        val.backward()
        return val, x.grad


def init_lipschitz(closure, x0, n_it=100):
    """Estimates the Lipschitz constant of obj_fun at x0
    in direction grad
    using backtracking line-search."""

    Lt = 1e-3
    f0, grad = closure(x0)

    xt = x0 - (1. / Lt) * grad

    ft = closure(xt, return_gradient=False)

    for _ in range(n_it):
        if (ft <= f0).all():
            break
        Lt *= 10
        xt = x0 - grad / Lt
        ft = closure(xt, return_gradient=False)
    return Lt
