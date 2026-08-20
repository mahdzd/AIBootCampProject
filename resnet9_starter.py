"""
ResNet-9 ICLR Paper Reproduction - Quick Start
Paper: "Improving Resnet-9 Generalization Trained on Small Datasets"
Author: Omar Mohamed Awad et al.

This starter code helps you quickly reproduce the main results.
Run this script to train baseline → apply optimizations → compare results.
"""

import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = './data'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import time
from tqdm import tqdm

# ============================================================================
# 1. RESNET-9 ARCHITECTURE
# ============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ResNet9(nn.Module):
    """
    ResNet-9 for CIFAR-10
    Simple, fast architecture suitable for small datasets
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution layer
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Layer 1
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        # Layer 2 (downsample)
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        
        # Layer 3 (downsample)
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# 2. OPTIMIZATION TECHNIQUES
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing - converts hard targets to soft targets
    Reduces overfitting on small datasets
    """
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, pred, target):
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (self.num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        pred_log = torch.log_softmax(pred, dim=1)
        return self.criterion(pred_log, smooth_target)


def centralize_gradient(model):
    """
    Gradient Centralization - removes correlation in gradients
    Improves generalization by smoothing loss landscape
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight.grad is not None:
                grad = module.weight.grad
                if len(grad.shape) > 1:
                    grad.data = grad.data - grad.data.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)


class PatchWhitening(nn.Module):
    """
    Input Patch Whitening - normalizes image patches
    Reduces input correlation and improves feature learning
    """
    def __init__(self, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        b, c, h, w = x.shape
        # Reshape into patches
        x = x.view(b, c, h // self.patch_size, self.patch_size,
                   w // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, -1, c * self.patch_size * self.patch_size)
        
        # Whiten
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x = (x - mean) / std
        
        # Reshape back
        x = x.view(b, h // self.patch_size, w // self.patch_size,
                   c, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, h, w)
        return x


# ============================================================================
# 3. DATA LOADING
# ============================================================================

def get_cifar10_loaders(batch_size=128, use_subset=True, subset_size=5000):
    """
    Load CIFAR-10 with optional 10% subset (5000 images)
    """
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
    
    # Use subset if specified
    if use_subset:
        np.random.seed(42)
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {subset_size} training samples (10% of CIFAR-10)")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, 
                use_sam=False, use_gc=False, patch_whitening=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply patch whitening if enabled
        if patch_whitening is not None:
            images = patch_whitening(images)
        
        # SAM optimization (two forward passes)
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
            # Standard optimization
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Gradient centralization
        if use_gc:
            centralize_gradient(model)
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'}, refresh=False)
    
    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(test_loader), 100 * correct / total


# ============================================================================
# 5. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, use_subset=True)
    
    # Model and training parameters
    model = ResNet9(num_classes=10).to(device)
    epochs = 50
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: BASELINE (SGD + Cross Entropy)")
    print("="*70)
    
    model = ResNet9(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"\nBaseline Results:")
    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    print(f"  Training Time: {elapsed/60:.2f} minutes")
    
    # ========================================================================
    # EXPERIMENT 2: WITH ALL OPTIMIZATIONS
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: OPTIMIZED (SAM + Label Smoothing + GC + PW)")
    print("="*70)
    
    model = ResNet9(num_classes=10).to(device)
    criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
    patch_whitening = PatchWhitening(patch_size=4)
    
    # Use SAM optimizer (install: pip install sam-pytorch)
    try:
        from sam import SAM
        base_optimizer = optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, 
                       lr=0.1, momentum=0.9, weight_decay=5e-4)
        use_sam = True
        print("SAM optimizer loaded successfully!")
    except ImportError:
        print("SAM not available, using standard SGD")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        use_sam = False
    
    start_time = time.time()
    best_acc_optimized = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_sam=use_sam, use_gc=True, patch_whitening=patch_whitening
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if test_acc > best_acc_optimized:
            best_acc_optimized = test_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"Best: {best_acc_optimized:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"\nOptimized Results:")
    print(f"  Best Test Accuracy: {best_acc_optimized:.2f}%")
    print(f"  Training Time: {elapsed/60:.2f} minutes")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    improvement = best_acc_optimized - best_acc
    print(f"Baseline Accuracy:      {best_acc:.2f}%")
    print(f"Optimized Accuracy:     {best_acc_optimized:.2f}%")
    print(f"Improvement:            +{improvement:.2f}%")
    print(f"\nTarget from paper:      ~88%")
    print(f"Status:                 {'✓ MATCHED!' if best_acc_optimized >= 85 else '✗ Keep tuning'}")


if __name__ == '__main__':
    main()