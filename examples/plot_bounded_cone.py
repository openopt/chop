"""
Bounded Cone optimization
==========================
In this example, we optimize a simple function over the intersection of a second order cone and a norm ball.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import chop

u = torch.tensor([[0, 0, 1.]])
cos_alpha = .5
cone = chop.constraints.Cone(u, cos_alpha)

norm_bound = chop.constraints.L2Ball(1.)


@chop.utils.closure
def obj_fun(x):
    return ((x-torch.tensor([[0, 0, 2.]])) ** 2).sum(dim=-1)


trace = chop.logging.Trace()

x0 = torch.rand(*u.shape)


res = chop.optim.minimize_three_split(obj_fun, x0, cone.prox, norm_bound.prox,
                                max_iter=100, callback=trace)

fig = plt.figure()
plt.plot([(fval - 1.) for fval in trace.trace_f])
plt.title("Function values")

# TODO: Plot the norm ball constraint and the second order cone constraint
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
points = [p.squeeze() for p in trace.trace_x]
xs, ys, zs = zip(*points)
ax.plot(xs, ys, zs)
plt.title("Iterates")

print(f"Final iterate: {res.x}\nFinal value: {res.fval}")
