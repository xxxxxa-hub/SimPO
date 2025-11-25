#!/usr/bin/env python3
"""
Analyze ICL pseudo-labeling results.

Usage:
    python analyze_pseudolabel_results.py <path_to_npz_file>
    python analyze_pseudolabel_results.py ./icl_pseudolabel_results/icl_results_k4.npz
"""

import numpy as np
import sys
import json
import os


def analyze_results(npz_file):
    """Load and analyze pseudo-labeling results."""

    print(f"Loading: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)

    # Load corresponding JSON file
    json_file = npz_file.replace('.npz', '.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results = json.load(f)
        has_json = True
    else:
        has_json = False
        print(f"Warning: JSON file not found: {json_file}")

    # Extract data
    predictions = data['predictions']
    confidences = data['confidences']
    ground_truth = data['ground_truth']
    pseudo_labels = data['pseudo_labels']
    high_conf_mask = data['high_conf_mask']
    demonstrations = data['demonstrations'].tolist()
    k_shot = int(data['k_shot'])
    conf_threshold = float(data['confidence_threshold'])

    print(f"\n{'='*80}")
    print(f"CONFIGURATION")
    print(f"{'='*80}")
    print(f"K-shot: {k_shot} examples per class ({2*k_shot} total demonstrations)")
    print(f"Test samples: {len(predictions)}")
    print(f"Confidence threshold: {conf_threshold}")

    # Overall metrics
    print(f"\n{'='*80}")
    print(f"OVERALL PERFORMANCE")
    print(f"{'='*80}")

    accuracy = (predictions == ground_truth).mean()
    print(f"Accuracy: {accuracy:.2%} ({(predictions == ground_truth).sum()}/{len(predictions)})")

    # Per-class accuracy
    for cls in [0, 1]:
        mask = ground_truth == cls
        if mask.sum() > 0:
            cls_acc = (predictions[mask] == cls).mean()
            cls_name = "objective" if cls == 0 else "subjective"
            print(f"  Class {cls} ({cls_name:11s}): {cls_acc:.2%} ({(predictions[mask] == cls).sum()}/{mask.sum()})")

    # Confidence analysis
    print(f"\n{'='*80}")
    print(f"CONFIDENCE ANALYSIS")
    print(f"{'='*80}")

    correct_mask = predictions == ground_truth
    incorrect_mask = ~correct_mask

    print(f"Overall confidence statistics:")
    print(f"  Mean: {confidences.mean():.3f}")
    print(f"  Std:  {confidences.std():.3f}")
    print(f"  Min:  {confidences.min():.3f}")
    print(f"  Max:  {confidences.max():.3f}")

    print(f"\nConfidence by correctness:")
    print(f"  Correct predictions:   mean={confidences[correct_mask].mean():.3f}, std={confidences[correct_mask].std():.3f}")
    print(f"  Incorrect predictions: mean={confidences[incorrect_mask].mean():.3f}, std={confidences[incorrect_mask].std():.3f}")

    # Pseudo-labeling results
    print(f"\n{'='*80}")
    print(f"PSEUDO-LABELING RESULTS")
    print(f"{'='*80}")

    num_high_conf = high_conf_mask.sum()
    pct_high_conf = 100 * num_high_conf / len(predictions)

    print(f"High-confidence predictions (≥{conf_threshold}): {num_high_conf}/{len(predictions)} ({pct_high_conf:.1f}%)")
    print(f"Low-confidence predictions (<{conf_threshold}): {(~high_conf_mask).sum()}/{len(predictions)} ({100-pct_high_conf:.1f}%)")

    if num_high_conf > 0:
        # Accuracy on high-confidence
        high_conf_correct = (predictions[high_conf_mask] == ground_truth[high_conf_mask]).sum()
        high_conf_acc = high_conf_correct / num_high_conf
        print(f"\nHigh-confidence accuracy: {high_conf_acc:.2%} ({high_conf_correct}/{num_high_conf})")

        # Distribution of pseudo-labels
        num_pseudo_0 = (pseudo_labels[high_conf_mask] == 0).sum()
        num_pseudo_1 = (pseudo_labels[high_conf_mask] == 1).sum()
        print(f"Pseudo-label distribution:")
        print(f"  Class 0 (objective):  {num_pseudo_0} ({100*num_pseudo_0/num_high_conf:.1f}%)")
        print(f"  Class 1 (subjective): {num_pseudo_1} ({100*num_pseudo_1/num_high_conf:.1f}%)")

        # Precision for each class (among high-confidence)
        for cls in [0, 1]:
            pseudo_cls_mask = (pseudo_labels == cls) & high_conf_mask
            if pseudo_cls_mask.sum() > 0:
                precision = (ground_truth[pseudo_cls_mask] == cls).mean()
                cls_name = "objective" if cls == 0 else "subjective"
                print(f"  Precision class {cls} ({cls_name}): {precision:.2%}")

    # Confidence threshold analysis
    print(f"\n{'='*80}")
    print(f"CONFIDENCE THRESHOLD SENSITIVITY")
    print(f"{'='*80}")
    print(f"{'Threshold':>10s} | {'Coverage':>10s} | {'Accuracy':>10s}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        mask = confidences >= thresh
        coverage = 100 * mask.sum() / len(predictions)
        acc = (predictions[mask] == ground_truth[mask]).mean() if mask.sum() > 0 else 0
        marker = " *" if thresh == conf_threshold else ""
        print(f"{thresh:>10.2f} | {coverage:>9.1f}% | {acc:>9.2%}{marker}")

    # Demonstrations
    print(f"\n{'='*80}")
    print(f"DEMONSTRATIONS")
    print(f"{'='*80}")

    for i, (text, label) in enumerate(demonstrations):
        label_name = "objective" if label == 0 else "subjective"
        # Truncate long text
        text_preview = text if len(text) <= 60 else text[:57] + "..."
        print(f"{i+1}. [{label_name:11s}] {text_preview}")

    # Error analysis
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS")
    print(f"{'='*80}")

    print(f"Total errors: {incorrect_mask.sum()}")

    # Show a few examples of errors
    error_indices = np.where(incorrect_mask)[0]
    if len(error_indices) > 0:
        print(f"\nSample errors (showing first 3):")
        for i, idx in enumerate(error_indices[:3]):
            print(f"\n  Error {i+1}:")
            print(f"    True label: {ground_truth[idx]} ({'objective' if ground_truth[idx]==0 else 'subjective'})")
            print(f"    Predicted:  {predictions[idx]} ({'objective' if predictions[idx]==0 else 'subjective'})")
            print(f"    Confidence: {confidences[idx]:.3f}")
            print(f"    High-conf:  {'Yes' if high_conf_mask[idx] else 'No'}")

    # Summary recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}")

    if pct_high_conf < 50:
        print(f"⚠ Low coverage ({pct_high_conf:.1f}%) - consider lowering confidence threshold")
    elif pct_high_conf > 95:
        print(f"⚠ Very high coverage ({pct_high_conf:.1f}%) - consider raising threshold for higher quality")
    else:
        print(f"✓ Good coverage ({pct_high_conf:.1f}%)")

    if num_high_conf > 0:
        if high_conf_acc < 0.85:
            print(f"⚠ Low pseudo-label accuracy ({high_conf_acc:.2%}) - model may not be well-calibrated")
        else:
            print(f"✓ Good pseudo-label accuracy ({high_conf_acc:.2%})")

    if accuracy < 0.75:
        print(f"⚠ Low overall accuracy ({accuracy:.2%}) - try increasing k-shot or using better demonstrations")
    else:
        print(f"✓ Good overall accuracy ({accuracy:.2%})")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pseudolabel_results.py <path_to_npz_file>")
        print("Example: python analyze_pseudolabel_results.py ./icl_pseudolabel_results/icl_results_k4.npz")
        sys.exit(1)

    npz_file = sys.argv[1]
    if not os.path.exists(npz_file):
        print(f"Error: File not found: {npz_file}")
        sys.exit(1)

    analyze_results(npz_file)
