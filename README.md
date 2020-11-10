# ConstOpt-PyTorch: a library for constrained optimization built on PyTorch
 ...with applications to Adversarially Training Neural Networks.

### Examples

Cf `examples/*` for MNIST and CIFAR10 examples. Results can be visualized using `tensorboard`.

### Tests

Run the tests with `pytests tests`. Then, run
```cox-tensorboard --logdir logging/tests/test_adversary --format-str alg-{algorithm}-step-size-{step-size}```
to visualize results using `cox` and `tensorboard`.

### Citing

If this software is useful to your research, please consider citing
```
@article{constopt-pytorch,
  author       = {Geoffrey Negiar, Fabian Pedregosa},
  title        = {constopt-pytorch: constrained optimization based on Pytorch},
  year         = 2020,
  url={http://github.com/openopt/constopt-pytorch}
}
```
