# pytorCH OPtimize: a library for continuous optimization built on PyTorch

...with applications to adversarially attacking and training neural networks.

[![Build Status](https://travis-ci.org/openopt/chop.svg?branch=master)](https://travis-ci.org/openopt/chop)
[![Coverage Status](https://coveralls.io/repos/github/openopt/chop/badge.svg?branch=master)](https://coveralls.io/github/openopt/chop?branch=master)
[![DOI](https://zenodo.org/badge/310693245.svg)](https://zenodo.org/badge/latestdoi/310693245)

:warning: This library is in early development, API might change without notice. The examples will be kept up to date. :warning:

## Stochastic Algorithms

We define stochastic optimizers in the `chop.stochastic` module. These follow PyTorch Optimizer conventions, similar to the `torch.optim` module.

## Full Gradient Algorithms

We also define full-gradient algorithms which operate on a batch of optimization problems in the `chop.optim` module. These are used for adversarial attacks, using the `chop.Adversary` wrapper.

## Examples:
  
  See `examples` directory and our [webpage](http://openo.pt/chop/auto_examples/index.html).

## Tests

Run the tests with `pytests tests`.

## Citing

If this software is useful to your research, please consider citing it as

```
@article{chop,
  author       = {Geoffrey Negiar, Fabian Pedregosa},
  title        = {CHOP: continuous optimization built on Pytorch},
  year         = 2020,
  url          = {http://github.com/openopt/chop}
}
```

## Affiliations

Geoffrey Négiar is in the Mahoney lab and the El Ghaoui lab at UC Berkeley.

Fabian Pedregosa is at Google Research.
