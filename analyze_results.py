#!/usr/bin/env python3
"""
Parse and analyze training results from logs
"""

import re
import json
from pathlib import Path


def parse_training_log():
    """Extract results from training log"""
    log_file = Path('logs/training.log')
    if not log_file.exists():
        print("Training log not found")
        return None

    results = {
        'baseline': {},
        'optimized': {},
        'comparison': {}
    }

    content = log_file.read_text()
    lines = content.split('\n')

    current_section = None
    best_acc_baseline = 0
    best_acc_optimized = 0

    for line in lines:
        # Detect section
        if 'BASELINE' in line:
            current_section = 'baseline'
        elif 'OPTIMIZED' in line:
            current_section = 'optimized'
        elif 'COMPARISON' in line:
            current_section = 'comparison'

        # Parse accuracy values
        if 'Best Test Accuracy:' in line:
            match = re.search(r'Best Test Accuracy: ([\d.]+)%', line)
            if match:
                acc = float(match.group(1))
                if current_section:
                    results[current_section]['best_accuracy'] = acc
                    if current_section == 'baseline':
                        best_acc_baseline = acc
                    elif current_section == 'optimized':
                        best_acc_optimized = acc

        if 'Training Time:' in line:
            match = re.search(r'Training Time: ([\d.]+) minutes', line)
            if match:
                time_min = float(match.group(1))
                if current_section:
                    results[current_section]['training_time'] = time_min

    if best_acc_baseline > 0 and best_acc_optimized > 0:
        results['comparison']['improvement'] = best_acc_optimized - best_acc_baseline
        results['comparison']['baseline_acc'] = best_acc_baseline
        results['comparison']['optimized_acc'] = best_acc_optimized

    return results


def create_summary_json():
    """Create results summary as JSON"""
    results = parse_training_log()
    if not results:
        return

    summary_data = {
        'experiment': 'ResNet-9 Small Dataset Generalization',
        'paper': 'ICLR 2023',
        'dataset': 'CIFAR-10 (10% subset - 5000 images)',
        'date': '2026-08-20',
        'results': results,
        'status': 'completed'
    }

    output_file = Path('logs/results_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"✓ Results summary saved to {output_file}")
    print(f"\nResults Summary:")
    print(f"  Baseline Accuracy: {results['baseline'].get('best_accuracy', 'N/A'):.2f}%")
    print(f"  Optimized Accuracy: {results['optimized'].get('best_accuracy', 'N/A'):.2f}%")
    print(f"  Improvement: +{results['comparison'].get('improvement', 0):.2f}%")
    print(f"  Paper Target: ~88%")


if __name__ == '__main__':
    create_summary_json()
