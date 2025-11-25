#!/usr/bin/env python3
"""
Iterative Context Optimization with Pseudo-Labeling for Binary Classification

This script implements an iterative context optimization algorithm similar to the pairwise
comparison version, but adapted for binary classification tasks.

Dataset: SetFit/subj (subjective vs objective text classification)

Algorithm:
1. Load dataset and split:
   - Test: 500 examples
   - Validation: 50 examples (no overlap with test)
   - Training: remaining examples
   - Demonstration pool: 10 random examples from training
2. Generate all k-shot permutations from 10 demonstrations (e.g., P(10,4) = 5040 for k=4)
3. Evaluate each permutation on validation set, compute SNR (signal-to-noise ratio)
4. Select top 3 contexts by SNR
5. For each top context, iteratively optimize by replacing demonstrations with test examples:
   - For each position, try n_candidates test examples as replacements
   - For each test example, try BOTH possible labels (objective and subjective)
   - Keep the (test_example, label) combination that gives highest validation SNR
6. Test both initial and optimized contexts on full test set
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import argparse
import os
from tqdm import tqdm
import json
import itertools
import random


def create_prompt(text, label=None, demonstrations=None):
    """
    Create ICL prompt for binary classification.

    Args:
        text: Input text to classify
        label: Ground truth label (0 or 1), optional
        demonstrations: List of (text, label) tuples for k-shot examples
    """
    # Label mapping for SetFit/subj dataset
    label_map = {0: "objective", 1: "subjective"}

    prompt = ""

    # Add demonstrations if provided
    if demonstrations:
        for i, (demo_text, demo_label) in enumerate(demonstrations):
            prompt += f"Sentence: {demo_text}\n"
            prompt += f"Answer: {label_map[demo_label]}\n\n"

    # Add the query
    prompt += f"Sentence: {text}\n"
    prompt += "Answer:"

    return prompt


def get_label_logits(model, tokenizer, prompt, device):
    """
    Get logits for 'objective' and 'subjective' tokens.

    Returns:
        log_probs: Log probabilities for [objective, subjective]
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits

    # Get token IDs for labels
    objective_id = tokenizer.encode(" objective", add_special_tokens=False)[0]
    subjective_id = tokenizer.encode(" subjective", add_special_tokens=False)[0]

    # Extract logits for these tokens
    label_logits = torch.stack([
        logits[:, objective_id],
        logits[:, subjective_id]
    ], dim=1)

    # Convert to log probabilities
    log_probs = F.log_softmax(label_logits, dim=1)

    return log_probs[0].cpu().numpy()


def sample_demonstration_pool(train_data, pool_size, seed=42):
    """
    Randomly sample a pool of demonstrations from training data.

    Args:
        train_data: Training dataset
        pool_size: Number of examples to sample (e.g., 10)
        seed: Random seed

    Returns:
        List of example dictionaries
    """
    np.random.seed(seed)
    indices = np.random.choice(len(train_data), pool_size, replace=False)
    return [train_data[int(i)] for i in indices]


def generate_k_shot_permutations(pool_size, k_shot):
    """
    Generate all k-shot permutations from a pool.

    Args:
        pool_size: Size of demonstration pool (e.g., 10)
        k_shot: Number of demonstrations to use (e.g., 4)

    Returns:
        List of tuples, each containing k indices
    """
    # Generate all permutations of k indices from pool
    return list(itertools.permutations(range(pool_size), k_shot))


def evaluate_context_on_validation(model, tokenizer, demo_pool, demo_indices,
                                   validation_set, device, demo_labels=None):
    """
    Evaluate a single context on validation set.

    Args:
        model: Language model
        tokenizer: Tokenizer
        demo_pool: Pool of demonstration examples
        demo_indices: Indices into demo_pool for this context (e.g., (0, 2, 5, 7))
        validation_set: Validation examples
        device: Device to run on
        demo_labels: Optional list of labels to use for each demo (overrides demo_pool labels).
                    Useful for test examples where we try both possible labels.

    Returns:
        log_probs_with_context: Log probabilities of correct class for each validation example
    """
    # Build demonstrations list
    if demo_labels is None:
        # Use labels from demo_pool
        demonstrations = [(demo_pool[i]['text'], demo_pool[i]['label']) for i in demo_indices]
    else:
        # Use provided labels (for trying different labels on test examples)
        demonstrations = [(demo_pool[i]['text'], demo_labels[j]) for j, i in enumerate(demo_indices)]

    log_probs_correct = []

    for val_example in validation_set:
        text = val_example['text']
        true_label = val_example['label']

        # Create prompt with demonstrations
        prompt = create_prompt(text, demonstrations=demonstrations)

        # Get predictions
        log_probs = get_label_logits(model, tokenizer, prompt, device)

        # Get log prob of correct class
        log_prob_correct = log_probs[true_label]
        log_probs_correct.append(log_prob_correct)

    return np.array(log_probs_correct)


def compute_no_context_baseline(model, tokenizer, validation_set, device):
    """
    Compute baseline log probabilities without context.

    Returns:
        log_probs_no_context: Log probabilities of correct class without demonstrations
    """
    log_probs_correct = []

    for val_example in validation_set:
        text = val_example['text']
        true_label = val_example['label']

        # Create prompt without demonstrations
        prompt = create_prompt(text, demonstrations=None)

        # Get predictions
        log_probs = get_label_logits(model, tokenizer, prompt, device)

        # Get log prob of correct class
        log_prob_correct = log_probs[true_label]
        log_probs_correct.append(log_prob_correct)

    return np.array(log_probs_correct)


def compute_snr(log_probs_with_context, log_probs_no_context):
    """
    Compute SNR (signal-to-noise ratio) of gain.

    Args:
        log_probs_with_context: Log probabilities with context
        log_probs_no_context: Log probabilities without context (baseline)

    Returns:
        snr: Signal-to-noise ratio (mean_gain / std_gain)
    """
    gain = log_probs_with_context - log_probs_no_context
    mean_gain = gain.mean()
    std_gain = gain.std()
    snr = mean_gain / (std_gain + 1e-10)  # Add small epsilon to avoid division by zero
    return snr


def run_icl_inference(model, tokenizer, test_data, demonstrations, device, batch_size=1):
    """
    Run ICL inference on test set.

    Returns:
        predictions: Predicted labels (0 or 1)
        log_probs_all: Log probabilities for both classes
    """
    predictions = []
    log_probs_all = []

    for example in tqdm(test_data, desc="Running ICL inference"):
        text = example['text']

        # Create prompt with demonstrations
        prompt = create_prompt(text, demonstrations=demonstrations)

        # Get predictions
        log_probs = get_label_logits(model, tokenizer, prompt, device)
        probs = np.exp(log_probs)

        # Predict class with higher probability
        pred_label = np.argmax(probs)

        predictions.append(pred_label)
        log_probs_all.append(log_probs)

    return np.array(predictions), np.array(log_probs_all)


def evaluate_predictions(predictions, ground_truth):
    """Calculate accuracy and other metrics."""
    accuracy = (predictions == ground_truth).mean()

    # Per-class accuracy
    mask_0 = ground_truth == 0
    mask_1 = ground_truth == 1

    acc_0 = (predictions[mask_0] == 0).mean() if mask_0.sum() > 0 else 0
    acc_1 = (predictions[mask_1] == 1).mean() if mask_1.sum() > 0 else 0

    return {
        'accuracy': accuracy,
        'accuracy_class_0': acc_0,
        'accuracy_class_1': acc_1,
        'num_samples': len(predictions)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative context optimization with pseudo-labeling")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name")
    parser.add_argument("--k_shot", type=int, default=4,
                       help="Number of demonstrations to use (k)")
    parser.add_argument("--pool_size", type=int, default=10,
                       help="Size of demonstration pool to sample from training")
    parser.add_argument("--test_size", type=int, default=500,
                       help="Number of test examples to evaluate")
    parser.add_argument("--val_size", type=int, default=50,
                       help="Number of validation examples")
    parser.add_argument("--top_n", type=int, default=3,
                       help="Number of best contexts to select and optimize")
    parser.add_argument("--n_candidates", type=int, default=100,
                       help="Number of test examples to try as replacements")
    parser.add_argument("--output_dir", type=str, default="./icl_optimized_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"{'='*80}")
    print(f"Iterative Context Optimization with Pseudo-Labeling")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"K-shot: {args.k_shot}")
    print(f"Demonstration pool size: {args.pool_size}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.val_size}")
    print(f"Top N contexts: {args.top_n}")
    print(f"N candidates for replacement: {args.n_candidates}")

    # Load model and tokenizer
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print(f"Model loaded on {device}")

    # Load dataset
    print(f"\nLoading SetFit/subj dataset...")
    dataset = load_dataset("SetFit/subj")

    # Combine train and test splits
    all_data = list(dataset['train']) + list(dataset['test'])
    print(f"Total dataset size: {len(all_data)}")

    # Split data: test (500), validation (50), training (remaining)
    random.shuffle(all_data)
    test_data = all_data[:args.test_size]
    validation_data = all_data[args.test_size:args.test_size + args.val_size]
    train_data = all_data[args.test_size + args.val_size:]

    print(f"Test set: {len(test_data)} examples")
    print(f"Validation set: {len(validation_data)} examples")
    print(f"Training set: {len(train_data)} examples")

    # Sample demonstration pool from training
    print(f"\nSampling {args.pool_size} examples as demonstration pool...")
    demo_pool = sample_demonstration_pool(train_data, args.pool_size, seed=args.seed)

    print(f"Demonstration pool:")
    for i, ex in enumerate(demo_pool):
        label_name = "objective" if ex['label'] == 0 else "subjective"
        print(f"  {i}. [{label_name}] {ex['text'][:80]}...")

    # Generate all k-shot permutations
    print(f"\n{'='*80}")
    print(f"Step 1: Generate all {args.k_shot}-shot permutations from {args.pool_size} demonstrations")
    print(f"{'='*80}")

    all_permutations = generate_k_shot_permutations(args.pool_size, args.k_shot)
    print(f"Total permutations: {len(all_permutations)} (P({args.pool_size},{args.k_shot}))")

    # Compute no-context baseline on validation set
    print(f"\nComputing no-context baseline on validation set...")
    breakpoint()
    log_probs_no_context = compute_no_context_baseline(model, tokenizer, validation_data, device)

    # Evaluate all permutations on validation set
    print(f"\n{'='*80}")
    print(f"Step 2: Evaluate all permutations on validation set and compute SNR")
    print(f"{'='*80}")

    snr_scores = []
    for i, perm_indices in enumerate(tqdm(all_permutations, desc="Evaluating permutations")):
        log_probs_with_context = evaluate_context_on_validation(
            model, tokenizer, demo_pool, perm_indices, validation_data, device
        )
        snr = compute_snr(log_probs_with_context, log_probs_no_context)
        snr_scores.append(snr)

    snr_scores = np.array(snr_scores)

    # Select top N contexts by SNR
    print(f"\n{'='*80}")
    print(f"Step 3: Select top {args.top_n} contexts by SNR")
    print(f"{'='*80}")

    top_indices = np.argsort(snr_scores)[::-1][:args.top_n]

    initial_contexts = []
    for i, idx in enumerate(top_indices):
        perm_indices = all_permutations[idx]
        snr = snr_scores[idx]
        initial_contexts.append(list(perm_indices))

        print(f"\nContext {i+1} (rank {i+1}):")
        print(f"  Demo indices: {perm_indices}")
        print(f"  SNR: {snr:.4f}")
        print(f"  Demonstrations:")
        for j, demo_idx in enumerate(perm_indices):
            ex = demo_pool[demo_idx]
            label_name = "objective" if ex['label'] == 0 else "subjective"
            print(f"    {j}. [{label_name}] {ex['text'][:60]}...")

    # Create candidate examples: demo_pool + test examples
    print(f"\n{'='*80}")
    print(f"Step 4: Prepare test examples as replacement candidates")
    print(f"{'='*80}")

    # Select random subset of test examples
    test_subset_indices = random.sample(range(len(test_data)), min(args.n_candidates, len(test_data)))
    test_subset = [test_data[i] for i in test_subset_indices]

    # For test examples, we don't know the correct label, so we keep them without labels
    # We'll try both possible labels (0 and 1) during optimization
    print(f"Selected {len(test_subset)} test examples as replacement candidates")
    print(f"Note: For each test example, we'll try BOTH labels (0 and 1) and keep the best")

    # Candidate examples: original demo_pool (0 to pool_size-1) + test subset (pool_size onwards)
    candidate_examples = demo_pool + test_subset
    test_offset = args.pool_size

    print(f"Total candidate pool size: {len(candidate_examples)}")
    print(f"  Original demo pool: indices 0-{args.pool_size-1}")
    print(f"  Test subset: indices {test_offset}-{len(candidate_examples)-1}")

    # Iterative optimization
    print(f"\n{'='*80}")
    print(f"Step 5: Iterative optimization - replace demonstrations with pseudo-labeled test examples")
    print(f"{'='*80}")

    optimized_contexts = []
    optimized_labels = []  # Track which labels we chose for each position

    for ctx_num, initial_indices in enumerate(initial_contexts):
        print(f"\n{'='*80}")
        print(f"Optimizing context {ctx_num + 1}/{args.top_n}")
        print(f"{'='*80}")
        print(f"Initial demo indices: {initial_indices}")

        # Get initial labels from demo_pool
        initial_labels = [demo_pool[i]['label'] for i in initial_indices]

        # Evaluate initial context
        initial_log_probs = evaluate_context_on_validation(
            model, tokenizer, candidate_examples, initial_indices, validation_data, device,
            demo_labels=initial_labels
        )
        initial_score = compute_snr(initial_log_probs, log_probs_no_context)
        print(f"Initial validation SNR: {initial_score:.4f}")

        current_indices = initial_indices.copy()
        current_labels = initial_labels.copy()
        current_score = initial_score

        # Optimize each position
        for pos in range(args.k_shot):
            print(f"\n  Optimizing position {pos}...")

            best_replacement_idx = current_indices[pos]
            best_replacement_label = current_labels[pos]
            best_replacement_score = current_score

            # Try test examples as replacements
            for test_idx in tqdm(range(test_offset, len(candidate_examples)),
                                desc=f"    Trying replacements at position {pos}",
                                leave=False):

                # For each test example, try BOTH possible labels (0 and 1)
                for candidate_label in [0, 1]:
                    candidate_indices = current_indices.copy()
                    candidate_labels = current_labels.copy()
                    candidate_indices[pos] = test_idx
                    candidate_labels[pos] = candidate_label

                    # Evaluate candidate with this label
                    candidate_log_probs = evaluate_context_on_validation(
                        model, tokenizer, candidate_examples, candidate_indices,
                        validation_data, device, demo_labels=candidate_labels
                    )
                    candidate_score = compute_snr(candidate_log_probs, log_probs_no_context)

                    if candidate_score > best_replacement_score:
                        best_replacement_idx = test_idx
                        best_replacement_label = candidate_label
                        best_replacement_score = candidate_score

            # Update if improvement found
            if best_replacement_score > current_score:
                label_name = "objective" if best_replacement_label == 0 else "subjective"
                print(f"    Found improvement! SNR: {current_score:.4f} -> {best_replacement_score:.4f}")
                print(f"    Replaced index {current_indices[pos]} with {best_replacement_idx} (label={label_name})")
                current_indices[pos] = best_replacement_idx
                current_labels[pos] = best_replacement_label
                current_score = best_replacement_score
            else:
                print(f"    No improvement found, keeping position {pos} unchanged")

        print(f"\nOptimization complete for context {ctx_num + 1}")
        print(f"  Initial: indices={initial_indices}, labels={initial_labels}, SNR={initial_score:.4f}")
        print(f"  Final:   indices={current_indices}, labels={current_labels}, SNR={current_score:.4f}")
        print(f"  SNR improvement: {current_score - initial_score:.4f}")

        optimized_contexts.append(current_indices)
        optimized_labels.append(current_labels)

    # Test initial and optimized contexts on full test set
    print(f"\n{'='*80}")
    print(f"Step 6: Testing contexts on full test set")
    print(f"{'='*80}")

    # Get ground truth labels
    ground_truth = np.array([ex['label'] for ex in test_data])

    # Compute no-context baseline on test set
    print(f"Computing no-context baseline on test set...")
    log_probs_no_context_test = compute_no_context_baseline(model, tokenizer, test_data, device)
    probs_no_context_test = np.exp(log_probs_no_context_test)
    accuracy_no_context = (probs_no_context_test > 0.5).astype(float).mean()

    print(f"No-context baseline accuracy: {accuracy_no_context:.4f}")

    # Test initial contexts
    print(f"\nTesting initial contexts on full test set...")
    initial_accuracies = []
    initial_context_labels = []

    for i, context_indices in enumerate(initial_contexts):
        # Get labels from original demo_pool
        context_labels = [demo_pool[idx]['label'] for idx in context_indices]
        initial_context_labels.append(context_labels)

        # Build demonstrations
        demonstrations = [(candidate_examples[idx]['text'], context_labels[j])
                         for j, idx in enumerate(context_indices)]

        # Run inference
        predictions, _ = run_icl_inference(model, tokenizer, test_data, demonstrations, device)

        # Compute accuracy
        accuracy = (predictions == ground_truth).mean()
        initial_accuracies.append(accuracy)

        print(f"  Context {i+1}: accuracy = {accuracy:.4f}, gain = {accuracy - accuracy_no_context:.4f}")

    # Test optimized contexts
    print(f"\nTesting optimized contexts on full test set...")
    optimized_accuracies = []

    for i, (context_indices, context_labels) in enumerate(zip(optimized_contexts, optimized_labels)):
        # Build demonstrations using optimized labels
        demonstrations = [(candidate_examples[idx]['text'], context_labels[j])
                         for j, idx in enumerate(context_indices)]

        # Run inference
        predictions, _ = run_icl_inference(model, tokenizer, test_data, demonstrations, device)

        # Compute accuracy
        accuracy = (predictions == ground_truth).mean()
        optimized_accuracies.append(accuracy)

        print(f"  Context {i+1}: accuracy = {accuracy:.4f}, gain = {accuracy - accuracy_no_context:.4f}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")

    for i in range(args.top_n):
        print(f"\nContext {i+1}:")
        print(f"  Initial (from demo pool):")
        print(f"    indices={initial_contexts[i]}")
        print(f"    labels={initial_context_labels[i]}")
        print(f"    accuracy={initial_accuracies[i]:.4f}")
        print(f"  Optimized (with pseudo-labels):")
        print(f"    indices={optimized_contexts[i]}")
        print(f"    labels={optimized_labels[i]}")
        print(f"    accuracy={optimized_accuracies[i]:.4f}")
        print(f"  Improvement: {optimized_accuracies[i] - initial_accuracies[i]:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"optimized_contexts_k{args.k_shot}.npz")

    np.savez(
        output_file,
        initial_contexts=initial_contexts,
        initial_context_labels=initial_context_labels,
        optimized_contexts=optimized_contexts,
        optimized_labels=optimized_labels,
        initial_accuracies=initial_accuracies,
        optimized_accuracies=optimized_accuracies,
        accuracy_no_context=accuracy_no_context,
        test_subset_indices=test_subset_indices,
        demo_pool=[{'text': ex['text'], 'label': int(ex['label'])} for ex in demo_pool],
        k_shot=args.k_shot,
        pool_size=args.pool_size,
        top_n=args.top_n
    )

    print(f"\nResults saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
