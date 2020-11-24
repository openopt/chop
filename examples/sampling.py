import torch
import matplotlib.pyplot as plt

from chop import utils
import chop


@utils.closure
def loss(x):
    return ((x - 1) ** 2).sum(dim=-1)
    mask = x[:, 0] > -x[:, 1]
    ret = torch.zeros(x.size(0), device=x.device)
    if mask.any():
        ret[mask] = .5 * ((x - 1) ** 2)[mask].sum(dim=-1)
    if (~mask).any():
        ret[~mask] = .5 * ((x + 1 ** 2))[~mask].sum(dim=-1)
    return ret


if __name__ == "__main__":
    batch_size = 1
    n_features = 2
    x0 = torch.normal(torch.zeros(batch_size, n_features))

    samples = []
    losses = []
    k = 0
    for sample in chop.sampling.sample_langevin(loss, x0, 1.):
        if k == 10000:
            break
        samples.append(sample.squeeze().detach().numpy())
        losses.append(loss(sample, return_jac=False))
        k += 1

    plt.scatter(*zip(*samples))
    # plt.ylim(-10, 10)
    # plt.xlim(-10, 10)
    plt.savefig('examples/plots/sampling/langevin.png')
