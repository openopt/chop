"""
Data loading utilities. 
=======================

Utility functions for data loading.
"""

from easydict import EasyDict

import torch
import torchvision
from torchvision import transforms




def load_cifar10(train_batch_size=128, test_batch_size=128, data_dir='~/datasets', augment_train=False):
    """Load training and test data."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                 transform=transform_train if augment_train else transform_test,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                transform=transform_test, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return EasyDict(train=train_loader, test=test_loader)
