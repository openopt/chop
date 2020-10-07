import torch
from torch.autograd import Variable
import numpy as np


class Adversary:
    def __init__(self, shape, constraint, optimizer_class):
        self.delta = Variable(torch.zeros(shape), requires_grad=True)
        self.optimizer = optimizer_class([self.delta], constraint)
        self.constraint = constraint

    def perturb(self, data, target, model, criterion,
                step_size, tol=1e-3, iterations=None,
                store=None):
        model.eval()
        loss = -criterion(model(data + self.delta), target)
        gap = torch.tensor(np.inf)
        ii = 0
        while gap.item() > tol:
            if ii == iterations:
                break
            if store:
                store.log_table_and_tb(self.optimizer.name,
                                       {'func_val': -loss.item(),
                                        'FW gap': gap.item(),
                                        'norm delta': (self.delta ** 2).sum()
                                        })
                store[self.optimizer.name].flush_row()
            loss.backward()
            with torch.no_grad():
                gap = self.constraint.fw_gap(self.delta.grad, self.delta)
            self.optimizer.step(step_size)
            loss = -criterion(model(data + self.delta), target)
            ii += 1
        return loss, self.delta
