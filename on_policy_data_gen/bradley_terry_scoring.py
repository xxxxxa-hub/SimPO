#!/usr/bin/env python3
"""
Estimate Bradley-Terry model scores from pairwise preference matrices.

The Bradley-Terry model estimates a strength parameter (score) for each item
such that the probability of item i beating item j is:
    P(i beats j) = score_i / (score_i + score_j)

This script reads preference matrices and uses iterative maximum likelihood
estimation to compute Bradley-Terry scores.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def bradley_terry_mle(preference_matrix: np.ndarray,
                      max_iter: int = 1000,
                      tol: float = 1e-6) -> np.ndarray:
    """
    Estimate Bradley-Terry scores using iterative MLE.

    Args:
        preference_matrix: NxN matrix where entry [i,j] is the probability
                          that item i is preferred over item j
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Array of Bradley-Terry scores (normalized so sum = n_items)
    """
    n_items = len(preference_matrix)

    # Replace NaN with 0.5 (no preference) for diagonal elements
    pref = np.copy(preference_matrix)
    np.fill_diagonal(pref, 0.5)

    # Count number of comparisons and wins
    # wins[i] = sum of all times i was preferred over j
    # comparisons[i] = sum of all comparisons involving i
    wins = np.nansum(pref, axis=1)  # Sum across columns (i vs all j)
    n_comparisons = np.sum(~np.isnan(preference_matrix), axis=1)

    # Initialize scores uniformly
    scores = np.ones(n_items)

    for iteration in range(max_iter):
        old_scores = scores.copy()

        # Update each score using the MM algorithm
        for i in range(n_items):
            denominator = 0.0
            for j in range(n_items):
                if i != j and not np.isnan(preference_matrix[i, j]):
                    # Add contribution from comparison with j
                    denominator += 1.0 / (scores[i] + scores[j])

            if denominator > 0:
                # Number of times i won
                wins_i = np.nansum(pref[i, :]) + np.nansum(pref[:, i] == 0) * 0
                # Actually count wins properly from both directions
                wins_i = np.nansum(pref[i, :])  # i preferred over others

                scores[i] = wins_i / denominator

        # Normalize to prevent overflow and maintain scale
        scores = scores / np.mean(scores) * n_items / n_items

        # Check convergence
        if np.max(np.abs(scores - old_scores)) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

    # Final normalization
    scores = scores / np.sum(scores) * n_items

    return scores


def bradley_terry_from_counts(win_matrix: np.ndarray,
                               max_iter: int = 1000,
                               tol: float = 1e-6) -> np.ndarray:
    """
    Estimate Bradley-Terry scores from win count matrix.

    Args:
        win_matrix: NxN matrix where entry [i,j] is number of times
                    i beat j (can be fractional for aggregated preferences)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Array of Bradley-Terry scores
    """
    n_items = len(win_matrix)
    scores = np.ones(n_items)

    # Total wins for each item
    wins = np.sum(win_matrix, axis=1)

    for iteration in range(max_iter):
        old_scores = scores.copy()

        for i in range(n_items):
            denominator = 0.0
            for j in range(n_items):
                if i != j:
                    n_ij = win_matrix[i, j] + win_matrix[j, i]
                    if n_ij > 0:
                        denominator += n_ij / (scores[i] + scores[j])

            if denominator > 0:
                scores[i] = wins[i] / denominator

        # Normalize
        scores = scores / np.mean(scores)

        if np.max(np.abs(scores - old_scores)) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

    return scores


def process_preference_data(input_file: str, output_file: str):
    """
    Process JSON file with preference matrices and compute Bradley-Terry scores.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    print(f"Loading data from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = []

    for idx, item in enumerate(data):
        sample_id = item.get('sample_id', idx)
        preference_matrix = np.array(item['preference_matrix'])

        # Compute Bradley-Terry scores
        bt_scores = bradley_terry_mle(preference_matrix)

        # Rank responses by score
        rankings = np.argsort(-bt_scores)  # Descending order

        result = {
            'sample_id': sample_id,
            'prompt': item.get('prompt', ''),
            'bt_scores': bt_scores.tolist(),
            'rankings': rankings.tolist(),
            'best_response_idx': int(rankings[0]),
            'preference_matrix': preference_matrix.tolist()
        }

        # Include responses if available
        if 'responses' in item:
            result['responses'] = item['responses']
            result['best_response'] = item['responses'][rankings[0]]

        results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} samples")

    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessed {len(results)} samples")
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Estimate Bradley-Terry scores from preference matrices'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Input JSON file with preference matrices'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file (default: input_file with _bt_scores suffix)'
    )

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_file)
        output_file = str(input_path.with_stem(input_path.stem + '_bt_scores'))
    else:
        output_file = args.output

    process_preference_data(args.input_file, output_file)


if __name__ == '__main__':
    main()