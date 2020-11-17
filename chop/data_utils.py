from easydict import EasyDict

import torch
import torchvision
from torchvision import transforms


def ld_cifar10():
    """Load training and test data."""
    transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root='~/datasets/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='~/datasets/', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    return EasyDict(train=train_loader, test=test_loader)
