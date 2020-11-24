"""
Contains Langevin-dynamics type algorithms for sampling.
All of these functions return iterators over the samples.
These functions operate batch_wise.
"""
import torch
import numpy as np

from chop import utils


def sample_langevin(closure, x0, step_size):
    if type(step_size) == float:
        step_size = step_size * torch.ones(x0.size(0), device=x0.device)
    xt = x0.detach().clone()
    batch_size = x0.size(0)
    mean = torch.zeros_like(xt, device=xt.device)
    std = 1.
    prev_loss = np.inf * torch.ones(batch_size)
    while True:
        loss, grad = closure(xt)
        with torch.no_grad():
            noise = torch.normal(mean=mean, std=std)
            # rejection sampling mask
            mask = loss < prev_loss
            xt_new = xt.detach().clone()
            xt_new[mask] = xt[mask] - utils.bmul(step_size[mask], grad[mask]) + utils.bmul(torch.sqrt(step_size[mask]), noise[mask])
            prev_loss[mask] = loss[mask].detach().clone()
            yield xt_new
            xt = xt_new.detach().clone()
