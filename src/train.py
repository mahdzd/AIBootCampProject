import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def train_epoch(model, train_loader, criterion, optimizer, device,
                use_sam=False, use_gc=False, patch_whitening=None):
    from .optimizers import centralize_gradient

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if patch_whitening is not None:
            images = patch_whitening(images)

        if use_sam:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if use_gc:
            centralize_gradient(model)

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item():.4f}, refresh=False)

    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(test_loader), 100 * correct / total


def evaluate_with_perturbations(model, test_loader, criterion, device,
                                noise_type='gaussian', noise_level=0.1):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images_orig = images.clone()
            images = images.to(device)
            labels = labels.to(device)

            if noise_type == 'gaussian':
                noise = torch.randn_like(images) * noise_level
                images = torch.clamp(images + noise, 0, 1)
            elif noise_type == 'salt_pepper':
                mask = torch.rand_like(images) < noise_level
                images[mask] = torch.where(
                    torch.rand_like(images) < 0.5,
                    torch.ones_like(images),
                    torch.zeros_like(images)
                )[mask]
            elif noise_type == 'brightness':
                images = images * (1 + (torch.rand(1).item() - 0.5) * noise_level)
                images = torch.clamp(images, 0, 1)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total
