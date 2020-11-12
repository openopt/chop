# ConstOpt-PyTorch: a library for constrained optimization built on PyTorch
 ...with applications to adversarially attacking and training neural networks.

## Examples

### Stochastic Constrained Algorithms
We define stochastic optimizers in the `constopt.stochastic` module. These follow PyTorch Optimizer conventions, similar to the `torch.optim` module.

Examples:
- `examples/training_constrained_net_on_mnist.py` for a model training use case.

### Full Gradient Constrained Algorithms

We also define full-gradient constrained algorithms which operate on a batch of optimization problems in the `constopt.optim` module. These are used for adversarial attacks, using the `constopt.Adversary` wrapper.

Examples:

- `examples/optim_dynamics.py` for a generic example (one datapoint in the batch)
- `examples/adversarial_robustness/attack_benchmark.py` for how to use our algorithms for adversarial attacks. 

## Tests

Run the tests with `pytests tests`.

## Citing

If this software is useful to your research, please consider citing
```
@article{constopt-pytorch,
  author       = {Geoffrey Negiar, Fabian Pedregosa},
  title        = {constopt-pytorch: constrained optimization based on Pytorch},
  year         = 2020,
  url          = {http://github.com/openopt/constopt-pytorch}
}
```
