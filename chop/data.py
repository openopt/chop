"""
Data loading utilities. 
=======================

Utility functions for data loading.
"""

from easydict import EasyDict

import torch
import torchvision
from torchvision import transforms


def load_cifar10(train_batch_size=128, test_batch_size=128, data_dir='~/datasets'):
    """Load training and test data."""
    transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return EasyDict(train=train_loader, test=test_loader)
