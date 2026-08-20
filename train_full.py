#!/usr/bin/env python3
"""
ResNet-9 Full Training & Evaluation Pipeline
Reproduces paper results + extension with robustness testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from pathlib import Path

from src import (
    ResNet9, LabelSmoothingLoss, centralize_gradient, PatchWhitening,
    get_cifar10_loaders, train_epoch, evaluate, evaluate_with_perturbations
)


def train_model(model, train_loader, test_loader, criterion, optimizer,
                device, epochs, use_sam=False, use_gc=False, patch_whitening=None):
    """Train model and return results"""
    best_acc = 0
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_sam=use_sam, use_gc=use_gc, patch_whitening=patch_whitening
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    return best_acc, train_accs, test_accs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, use_subset=True)

    results = {}

    # ========================================================================
    # EXPERIMENT 1: BASELINE (SGD + CrossEntropyLoss)
    # ========================================================================
    print("=" * 70)
    print("EXPERIMENT 1: BASELINE (SGD + Cross Entropy Loss)")
    print("=" * 70)

    model = ResNet9(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    start_time = time.time()
    best_acc_baseline, train_acc_baseline, test_acc_baseline = train_model(
        model, train_loader, test_loader, criterion, optimizer, device,
        epochs=50, use_sam=False, use_gc=False, patch_whitening=None
    )
    elapsed_baseline = time.time() - start_time

    print(f"\nBaseline Results:")
    print(f"  Best Test Accuracy: {best_acc_baseline:.2f}%")
    print(f"  Training Time: {elapsed_baseline/60:.2f} minutes\n")

    results['baseline'] = {
        'accuracy': best_acc_baseline,
        'time_minutes': elapsed_baseline / 60,
        'train_accs': train_acc_baseline,
        'test_accs': test_acc_baseline,
    }

    # ========================================================================
    # EXPERIMENT 2: OPTIMIZED (SAM + Label Smoothing + GC + Patch Whitening)
    # ========================================================================
    print("=" * 70)
    print("EXPERIMENT 2: OPTIMIZED (SAM + LS + GC + PW)")
    print("=" * 70)

    model = ResNet9(num_classes=10).to(device)
    criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
    patch_whitening = PatchWhitening(patch_size=4)

    use_sam = False
    try:
        from sam import SAM
        base_optimizer = optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer,
                       lr=0.1, momentum=0.9, weight_decay=5e-4)
        use_sam = True
        print("✓ SAM optimizer loaded successfully!")
    except ImportError:
        print("✗ SAM not available, using standard SGD instead")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    start_time = time.time()
    best_acc_optimized, train_acc_optimized, test_acc_optimized = train_model(
        model, train_loader, test_loader, criterion, optimizer, device,
        epochs=50, use_sam=use_sam, use_gc=True, patch_whitening=patch_whitening
    )
    elapsed_optimized = time.time() - start_time

    print(f"\nOptimized Results:")
    print(f"  Best Test Accuracy: {best_acc_optimized:.2f}%")
    print(f"  Training Time: {elapsed_optimized/60:.2f} minutes\n")

    results['optimized'] = {
        'accuracy': best_acc_optimized,
        'time_minutes': elapsed_optimized / 60,
        'train_accs': train_acc_optimized,
        'test_accs': test_acc_optimized,
    }

    # ========================================================================
    # EXPERIMENT 3: ROBUSTNESS TESTING (Extension)
    # ========================================================================
    print("=" * 70)
    print("EXPERIMENT 3: ROBUSTNESS TO PERTURBATIONS (Extension)")
    print("=" * 70)

    noise_configs = [
        ('gaussian', 0.05),
        ('gaussian', 0.1),
        ('gaussian', 0.2),
        ('salt_pepper', 0.1),
        ('brightness', 0.2),
    ]

    print("\nBaseline Model Robustness:")
    baseline_model = ResNet9(num_classes=10).to(device)
    baseline_robust = {}
    for noise_type, noise_level in noise_configs:
        acc = evaluate_with_perturbations(
            baseline_model, test_loader, criterion, device,
            noise_type=noise_type, noise_level=noise_level
        )
        key = f"{noise_type}_{noise_level}"
        baseline_robust[key] = acc
        print(f"  {noise_type} (level={noise_level}): {acc:.2f}%")

    print("\nOptimized Model Robustness:")
    optimized_robust = {}
    for noise_type, noise_level in noise_configs:
        acc = evaluate_with_perturbations(
            model, test_loader, criterion, device,
            noise_type=noise_type, noise_level=noise_level
        )
        key = f"{noise_type}_{noise_level}"
        optimized_robust[key] = acc
        print(f"  {noise_type} (level={noise_level}): {acc:.2f}%")

    results['robustness'] = {
        'baseline': baseline_robust,
        'optimized': optimized_robust,
    }

    # ========================================================================
    # COMPARISON & SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    improvement = best_acc_optimized - best_acc_baseline
    print(f"\nAccuracy:")
    print(f"  Baseline:       {best_acc_baseline:.2f}%")
    print(f"  Optimized:      {best_acc_optimized:.2f}%")
    print(f"  Improvement:    +{improvement:.2f}%")
    print(f"  Paper Target:   ~88%")

    status = "✓ MATCHED!" if best_acc_optimized >= 85 else "✗ Keep tuning"
    print(f"  Status:         {status}")

    print(f"\nTraining Time:")
    print(f"  Baseline:       {elapsed_baseline/60:.2f} min")
    print(f"  Optimized:      {elapsed_optimized/60:.2f} min")

    # Calculate average robustness improvement
    avg_baseline_robust = sum(baseline_robust.values()) / len(baseline_robust)
    avg_optimized_robust = sum(optimized_robust.values()) / len(optimized_robust)
    robust_improvement = avg_optimized_robust - avg_baseline_robust

    print(f"\nAverage Robustness (across noise types):")
    print(f"  Baseline:       {avg_baseline_robust:.2f}%")
    print(f"  Optimized:      {avg_optimized_robust:.2f}%")
    print(f"  Improvement:    +{robust_improvement:.2f}%")

    # Save results
    Path('logs').mkdir(exist_ok=True)
    with open('logs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to logs/results.json")

    return results


if __name__ == '__main__':
    main()
