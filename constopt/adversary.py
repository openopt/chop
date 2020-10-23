import torch
from torch.autograd import Variable
import numpy as np


class Adversary:
    def __init__(self, shape, constraint, optimizer_class,
                 device=None, random_init=False):
        if random_init:
            self.delta = Variable(constraint.random_point(shape))
        else:
            self.delta = Variable(torch.zeros(shape))
        self.delta = self.delta.to(device)
        self.delta.requires_grad = True
        self.optimizer = optimizer_class([self.delta], constraint) if optimizer_class else None
        self.constraint = constraint

    def perturb(self, data, target, model, criterion,
                step_size=None, tol=1e-6, iterations=None,
                use_best=False,
                store=None):

        if self.optimizer is None:
            return criterion(model(data), target), 0.
        was_training = model.training
        model.eval()
        ii = 0

        gap = torch.tensor(np.inf)
        best_loss = -torch.tensor(np.inf)
        while gap.item() > tol:
            if ii == iterations:
                break

            self.optimizer.zero_grad()
            output = model(data + self.delta)
            adv_loss = -criterion(output, target)
            if -adv_loss.item() > best_loss.item():
                best_loss = -adv_loss.clone().detach()
                best_delta = self.delta.clone().detach()
            adv_loss.backward()

            with torch.no_grad():
                gap = self.constraint.fw_gap(self.delta.grad, self.delta)

            self.optimizer.step(step_size, batch=True)
            ii += 1

            # Logging
            if store:
                # Might be better to use the same name for all optimizers, to get
                # only one plot
                def norm(x):
                    if self.constraint.p == 1:
                        return abs(x).sum()
                    if self.constraint.p == 2:
                        return torch.sqrt((x ** 2).sum())
                    if self.constraint.p == np.inf:
                        return abs(x).max()
                    raise NotImplementedError("We've only implemented p = 1, 2, np.inf")
                p = self.constraint.p
                table_name = "L" + str(int(p)) + " ball" if p != np.inf else "Linf Ball"
                store.log_table_and_tb(table_name,
                                       {'func_val': -adv_loss.item(),
                                        'FW gap': gap.item(),
                                        'norm delta': norm(self.delta)
                                        })
                store[table_name].flush_row()

        if was_training:
            model.train()

        if use_best:
            return -best_loss, best_delta

        return -adv_loss, self.delta
