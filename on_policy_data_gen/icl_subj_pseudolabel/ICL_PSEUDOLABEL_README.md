# Simple In-Context Learning with Pseudo-Labeling

A clean implementation for testing ICL-based pseudo-labeling on the SetFit/subj dataset.

## Dataset

**SetFit/subj**: Binary text classification (subjective vs objective)
- Training set: ~10,000 examples
- Test set: 10,000 examples
- Task: Classify text as subjective or objective

## What This Does

1. **Select k-shot demonstrations**: Balanced selection from training set
2. **Run ICL inference**: Use demonstrations to classify test examples
3. **Evaluate accuracy**: Compare predictions to ground truth
4. **Generate pseudo-labels**: Create labels for high-confidence predictions
5. **Save results**: Output predictions, confidences, and metrics

## Quick Start

### Basic Usage
```bash
bash run_simple_icl.sh
```

### Custom Parameters
```bash
python simple_icl_pseudolabel.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --k_shot 4 \
    --test_size 500 \
    --confidence_threshold 0.8 \
    --output_dir ./icl_pseudolabel_results \
    --seed 42
```

## Parameters

- `--model_name`: HuggingFace model name
- `--k_shot`: Number of examples **per class** (total demos = 2 × k_shot)
  - k_shot=4 → 8 demonstrations total (4 objective + 4 subjective)
- `--test_size`: Number of test examples to evaluate
- `--confidence_threshold`: Minimum confidence (probability) for pseudo-labeling
- `--output_dir`: Where to save results
- `--seed`: Random seed for reproducibility

## Output Files

### 1. `icl_results_k{k_shot}.npz` (Numpy format)
```python
import numpy as np
data = np.load('icl_pseudolabel_results/icl_results_k4.npz', allow_pickle=True)

# Available arrays:
predictions = data['predictions']          # Predicted labels (0 or 1)
confidences = data['confidences']          # Confidence scores [0, 1]
log_probs = data['log_probs']             # Log probabilities for both classes
ground_truth = data['ground_truth']       # True labels
pseudo_labels = data['pseudo_labels']     # Pseudo-labels (-1 for low confidence)
high_conf_mask = data['high_conf_mask']   # Boolean mask for high-confidence
demonstrations = data['demonstrations']    # Selected k-shot examples
```

### 2. `icl_results_k{k_shot}.json` (Human-readable)
```json
{
  "config": {...},
  "metrics": {
    "accuracy": 0.85,
    "accuracy_class_0": 0.83,
    "accuracy_class_1": 0.87
  },
  "pseudo_labeling": {
    "num_high_confidence": 425,
    "pct_high_confidence": 85.0,
    "high_conf_accuracy": 0.92
  },
  "demonstrations": [...]
}
```

## Understanding Pseudo-Labels

### What are pseudo-labels?
Predictions with confidence ≥ threshold are treated as "pseudo ground truth"

### How it works:
1. Model predicts: class=1, confidence=0.95 → **High confidence** → Pseudo-label = 1 ✓
2. Model predicts: class=0, confidence=0.65 → **Low confidence** → Pseudo-label = -1 (unlabeled)

### Use cases:
- **Semi-supervised learning**: Use pseudo-labels to train on unlabeled data
- **Active learning**: Focus labeling effort on low-confidence examples
- **Data filtering**: Keep only high-quality predictions

## Example Analysis

```python
import numpy as np
import json

# Load results
data = np.load('icl_pseudolabel_results/icl_results_k4.npz', allow_pickle=True)
with open('icl_pseudolabel_results/icl_results_k4.json', 'r') as f:
    results = json.load(f)

# Overall metrics
print(f"Accuracy: {results['metrics']['accuracy']:.2%}")

# Pseudo-labeling quality
pl = results['pseudo_labeling']
print(f"High-confidence examples: {pl['pct_high_confidence']:.1f}%")
print(f"Accuracy on high-conf: {pl['high_conf_accuracy']:.2%}")

# Analyze confidence distribution
predictions = data['predictions']
confidences = data['confidences']
ground_truth = data['ground_truth']

# Correct vs incorrect predictions
correct_mask = predictions == ground_truth
print(f"Mean confidence (correct): {confidences[correct_mask].mean():.3f}")
print(f"Mean confidence (incorrect): {confidences[~correct_mask].mean():.3f}")
```

## Experiment Ideas

### 1. **Vary k-shot**
```bash
for k in 1 2 4 8; do
    python simple_icl_pseudolabel.py --k_shot $k --test_size 500
done
```
Question: Does more demonstrations improve accuracy?

### 2. **Confidence threshold sweep**
```bash
for thresh in 0.5 0.6 0.7 0.8 0.9; do
    python simple_icl_pseudolabel.py \
        --k_shot 4 \
        --confidence_threshold $thresh
done
```
Question: What's the trade-off between coverage and accuracy?

### 3. **Pseudo-label quality vs quantity**
- Low threshold (e.g., 0.6): More pseudo-labels, lower accuracy
- High threshold (e.g., 0.9): Fewer pseudo-labels, higher accuracy

### 4. **Iterative pseudo-labeling**
1. Run ICL, get high-confidence pseudo-labels
2. Add pseudo-labeled examples to demonstration pool
3. Re-run ICL with expanded demonstrations
4. Repeat

## Next Steps

Based on results, you can:
1. **Use pseudo-labels for training**: Train a smaller model on pseudo-labeled data
2. **Improve demonstration selection**: Select demos based on similarity to test examples
3. **Calibrate confidence**: Adjust threshold based on validation set
4. **Multi-round labeling**: Iteratively expand labeled set with pseudo-labels

## Notes

- **SetFit/subj is balanced**: ~50% objective, ~50% subjective
- **Confidence ≠ Accuracy**: High confidence doesn't guarantee correctness
- **Demonstration order matters**: Try shuffling demos for robustness
- **Model-dependent**: Different models have different calibration properties
