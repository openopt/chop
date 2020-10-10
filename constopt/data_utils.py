from easydict import EasyDict

import torch
import torchvision


def ld_cifar10():
  """Load training and test data."""
  train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  train_dataset = torchvision.datasets.CIFAR10(root='~/datasets/', train=True, transform=train_transforms, download=True)
  test_dataset = torchvision.datasets.CIFAR10(root='~/datasets/', train=False, transform=test_transforms, download=True)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
  return EasyDict(train=train_loader, test=test_loader)