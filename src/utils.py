import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np


def get_cifar10_loaders(batch_size=128, use_subset=True, subset_size=5000):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    print("Downloading CIFAR-10...")
    train_dataset = CIFAR10('./data', train=True, download=True,
                            transform=transform_train)
    test_dataset = CIFAR10('./data', train=False, download=True,
                           transform=transform_test)

    if use_subset:
        np.random.seed(42)
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {subset_size} training samples (10% of CIFAR-10)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    return train_loader, test_loader
