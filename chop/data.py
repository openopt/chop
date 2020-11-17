import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms


def load_cifar10(batch_size=100, data_dir='./data'):
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(root=data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=batch_size, shuffle=False)

    return test_loader
