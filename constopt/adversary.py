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
        ii = 0
        if "FW" in self.optimizer.name:
            print("FW")
            step_size = 0.

        gap = torch.tensor(np.inf)
        while gap.item() > tol:
            if ii == iterations:
                break

            self.optimizer.zero_grad()
            output = model(data + self.delta)
            loss = -criterion(output, target)
            loss.backward()

            with torch.no_grad():
                gap = self.constraint.fw_gap(self.delta.grad, self.delta)
            if store:
                # Might be better to use the same name for all optimizers, to get
                # only one plot
                store.log_table_and_tb(self.optimizer.name,
                                    {'func_val': -loss.item(),
                                        'FW gap': gap.item(),
                                        'norm delta': abs(self.delta).sum()
                                        })
                store[self.optimizer.name].flush_row()

            self.optimizer.step(step_size)
            ii += 1

        return loss, self.delta
