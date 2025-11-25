# Simple In-Context Learning with Pseudo-Labeling

A clean implementation for testing ICL-based pseudo-labeling on the SetFit/subj dataset.

## Dataset

**SetFit/subj**: Binary text classification (subjective vs objective)
- Training set: ~10,000 examples
- Test set: 10,000 examples
- Task: Classify text as subjective or objective

## Files

- **`simple_icl_pseudolabel.py`** - Main ICL script with pseudo-labeling
- **`run_simple_icl.sh`** - Bash script to run experiments
- **`analyze_pseudolabel_results.py`** - Analysis tool for results
- **`ICL_PSEUDOLABEL_README.md`** - Complete documentation

## Quick Start

```bash
# Run with defaults (4-shot, 500 test examples)
bash run_simple_icl.sh

# Analyze results
python analyze_pseudolabel_results.py ./icl_pseudolabel_results/icl_results_k4.npz
```

## What It Does

1. **Select k-shot demonstrations**: Balanced selection from training set
2. **Run ICL inference**: Use demonstrations to classify test examples
3. **Evaluate accuracy**: Compare predictions to ground truth
4. **Generate pseudo-labels**: Create labels for high-confidence predictions
5. **Save results**: Output predictions, confidences, and metrics

## See Full Documentation

For complete documentation, see [ICL_PSEUDOLABEL_README.md](ICL_PSEUDOLABEL_README.md)
