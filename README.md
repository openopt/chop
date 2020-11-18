[![Build Status](https://travis-ci.org/openopt/chop.svg?branch=master)](https://travis-ci.org/openopt/chop)
# pytorCH OPtimize: a library for continuous optimization built on PyTorch
 ...with applications to adversarially attacking and training neural networks.
 
!! WARNING !! This library is in early development. Its API may change often for the time being.

## Stochastic Algorithms
We define stochastic optimizers in the `chop.stochastic` module. These follow PyTorch Optimizer conventions, similar to the `torch.optim` module.

### Examples:
- `examples/training_constrained_net_on_mnist.py` for a model training use case.

## Full Gradient Constrained Algorithms

We also define full-gradient constrained algorithms which operate on a batch of optimization problems in the `chop.optim` module. These are used for adversarial attacks, using the `chop.Adversary` wrapper.

### Examples:

- `examples/optim_dynamics.py` for a generic example (one datapoint in the batch)
- `examples/adversarial_robustness/attack_benchmark.py` for how to use our algorithms for adversarial attacks. 

## Tests

Run the tests with `pytests tests`.

## Citing

If this software is useful to your research, please consider citing
```
@article{chop,
  author       = {Geoffrey Negiar, Fabian Pedregosa},
  title        = {CHOP: continuous optimization built on Pytorch},
  year         = 2020,
  url          = {http://github.com/openopt/chop}
}
```
