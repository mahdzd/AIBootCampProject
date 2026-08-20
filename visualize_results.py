#!/usr/bin/env python3
"""
Visualization script for ResNet-9 training results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves():
    """Plot training and validation curves"""
    try:
        with open('logs/results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Results file not found. Run training first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training Accuracy
    axes[0].plot(results['baseline']['train_accs'], label='Baseline (Train)', marker='o', alpha=0.7)
    axes[0].plot(results['optimized']['train_accs'], label='Optimized (Train)', marker='s', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Test Accuracy
    axes[1].plot(results['baseline']['test_accs'], label='Baseline (Test)', marker='o', alpha=0.7)
    axes[1].plot(results['optimized']['test_accs'], label='Optimized (Test)', marker='s', alpha=0.7)
    axes[1].axhline(y=88, color='red', linestyle='--', label='Paper Target (88%)', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig('outputs/training_curves.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: outputs/training_curves.png")
    plt.close()


def plot_robustness_comparison():
    """Plot robustness to different noise types"""
    try:
        with open('logs/results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Results file not found. Run training first.")
        return

    robustness = results['robustness']
    baseline = robustness['baseline']
    optimized = robustness['optimized']

    noise_types = []
    baseline_accs = []
    optimized_accs = []

    for key in sorted(baseline.keys()):
        noise_types.append(key.replace('_', '\n'))
        baseline_accs.append(baseline[key])
        optimized_accs.append(optimized[key])

    x = np.arange(len(noise_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, optimized_accs, width, label='Optimized', alpha=0.8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustness to Different Perturbations')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig('outputs/robustness_comparison.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: outputs/robustness_comparison.png")
    plt.close()


def plot_accuracy_comparison():
    """Plot accuracy comparison"""
    try:
        with open('logs/results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Results file not found. Run training first.")
        return

    baseline_acc = results['baseline']['accuracy']
    optimized_acc = results['optimized']['accuracy']

    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['Baseline\n(SGD)', 'Optimized\n(SAM+LS+GC+PW)']
    accs = [baseline_acc, optimized_acc]
    colors = ['#3498db', '#2ecc71']

    bars = ax.bar(methods, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    ax.axhline(y=88, color='red', linestyle='--', linewidth=2, label='Paper Target (88%)', alpha=0.7)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('ResNet-9 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([70, 95])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, acc) in enumerate(zip(bars, accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig('outputs/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: outputs/accuracy_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("Generating visualizations...")
    plot_accuracy_comparison()
    plot_training_curves()
    plot_robustness_comparison()
    print("\n[OK] All visualizations generated successfully!")
