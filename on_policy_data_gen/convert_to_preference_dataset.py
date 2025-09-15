#!/usr/bin/env python3
"""
Convert pairwise preference comparison results to a preference dataset format.
Reads the JSON file with preference matrices, corrects the probabilities,
finds the most extreme preferences, and creates a dataset in the format of
princeton-nlp/gemma2-ultrafeedback-armorm.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from typing import Dict, List, Tuple, Optional

def correct_preference_matrix(matrix):
    """
    Correct the preference matrix by averaging complementary pairs.
    
    Raw matrix contains P(A>B) at [i,j] and P(B>A) at [j,i] from different prompts.
    Corrected P(A>B) = (P(A>B) + 1 - P(B>A)) / 2
    
    Args:
        matrix: Raw preference matrix
        
    Returns:
        corrected_matrix: Matrix with true preference probabilities
    """
    matrix = np.array(matrix)
    n = matrix.shape[0]
    corrected = np.full((n, n), np.nan)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                p_a_over_b = matrix[i, j]  # P(A>B) from prompt "A vs B"
                p_b_over_a = matrix[j, i]  # P(B>A) from prompt "B vs A"
                
                # Handle missing values
                if not np.isnan(p_a_over_b) and not np.isnan(p_b_over_a):
                    # True P(A>B) = (P(A>B) + 1 - P(B>A)) / 2
                    true_p_a_over_b = (p_a_over_b + 1 - p_b_over_a) / 2
                    corrected[i, j] = true_p_a_over_b
                elif not np.isnan(p_a_over_b):
                    # Only one measurement available, use as-is
                    corrected[i, j] = p_a_over_b
                elif not np.isnan(p_b_over_a):
                    # Use complement of available measurement
                    corrected[i, j] = 1 - p_b_over_a
    
    return corrected

def find_most_extreme_preference(matrix):
    """
    Find the pair with the most extreme preference (closest to 0 or 1).
    
    Args:
        matrix: Corrected preference matrix
        
    Returns:
        chosen_idx: Index of chosen response
        rejected_idx: Index of rejected response
        preference_prob: The extreme preference probability
        confidence: How extreme the preference is (distance from 0.5)
    """
    n = matrix.shape[0]
    max_confidence = 0
    best_chosen_idx = 0
    best_rejected_idx = 1
    best_prob = 0.5
    
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(matrix[i, j]):
                prob = matrix[i, j]
                confidence = abs(prob - 0.5)  # Distance from random
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    if prob > 0.5:
                        # i is preferred over j
                        best_chosen_idx = i
                        best_rejected_idx = j
                    else:
                        # j is preferred over i
                        best_chosen_idx = j
                        best_rejected_idx = i
                    best_prob = prob if prob > 0.5 else (1 - prob)
    
    return best_chosen_idx, best_rejected_idx, best_prob, max_confidence


def create_conversation_format(prompt: str, response: str) -> List[Dict[str, str]]:
    """Create the conversation format expected by the dataset."""
    return [
        {"content": prompt, "role": "user"},
        {"content": response, "role": "assistant"}
    ]

def convert_to_preference_dataset(input_files: Dict[str, str], output_dir: str = None) -> DatasetDict:
    """
    Convert preference matrix results to preference dataset format.

    Args:
        input_files: Dict mapping split names to JSON file paths with preference matrices
        output_dir: Optional directory to save the dataset

    Returns:
        DatasetDict object with splits in the target format
    """
    splits = {}
    total_samples = 0
    total_skipped = 0

    for split_name, input_file in input_files.items():
        print(f"\nProcessing {split_name} split from {input_file}")

        with open(input_file, 'r') as f:
            results = json.load(f)

        print(f"Loaded {len(results)} samples for {split_name}")

        dataset_entries = []
        skipped_samples = 0

        for sample in results:
            try:
                prompt = sample['prompt']
                responses = sample['responses']
                raw_matrix = sample['preference_matrix']

                # Correct the preference matrix
                corrected_matrix = correct_preference_matrix(raw_matrix)

                # Find the most extreme preference
                chosen_idx, rejected_idx, preference_prob, confidence = find_most_extreme_preference(corrected_matrix)

                # Create dataset entry with only essential columns
                entry = {
                    "prompt": prompt,
                    "chosen": create_conversation_format(prompt, responses[chosen_idx]),
                    "rejected": create_conversation_format(prompt, responses[rejected_idx]),
                }

                dataset_entries.append(entry)

            except Exception as e:
                print(f"Error processing sample {sample.get('sample_id', 'unknown')}: {e}")
                skipped_samples += 1
                continue

        print(f"Successfully processed {len(dataset_entries)} samples for {split_name}")
        print(f"Skipped {skipped_samples} samples for {split_name}")

        # Create dataset for this split
        splits[split_name] = Dataset.from_list(dataset_entries)
        total_samples += len(dataset_entries)
        total_skipped += skipped_samples

    print(f"\n=== Overall Statistics ===")
    print(f"Total samples across all splits: {total_samples}")
    print(f"Total skipped samples: {total_skipped}")

    # Create DatasetDict
    dataset_dict = DatasetDict(splits)

    # Save if output directory specified
    if output_dir:
        print(f"Saving dataset to {output_dir}")
        dataset_dict.save_to_disk(output_dir)

    return dataset_dict

def push_to_hub(dataset_dict: DatasetDict, repo_id: str,
               private: bool = False, commit_message: str = None):
    """
    Push the dataset to Hugging Face Hub.

    Args:
        dataset_dict: DatasetDict with splits to push
        repo_id: Repository ID (username/dataset-name)
        private: Whether to make the repository private
        commit_message: Commit message
    """
    if commit_message is None:
        commit_message = "Add preference dataset from corrected preference matrices"

    print(f"Pushing dataset with splits {list(dataset_dict.keys())} to {repo_id}")

    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message=commit_message
    )

    print(f"Successfully pushed dataset to https://huggingface.co/datasets/{repo_id}")

def create_dataset_card(repo_id: str, stats: Dict, original_dataset_info: str = None):
    """Create a dataset card for the uploaded dataset."""

    # Format split statistics
    split_stats = ""
    for split_name, count in stats.get('splits', {}).items():
        split_stats += f"- **{split_name.title()}**: {count} samples\n"

    dataset_card = f"""# {repo_id}

This dataset contains preference pairs extracted from pairwise preference comparison matrices, formatted similarly to [princeton-nlp/gemma2-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/gemma2-ultrafeedback-armorm).

## Dataset Creation

### Source Data
{original_dataset_info or "Generated from pairwise preference comparison experiments."}

### Processing Steps

1. **Preference Matrix Correction**: Raw preference probabilities P(A>B) and P(B>A) from different prompt orders were corrected using:
   ```
   Corrected P(A>B) = (Raw P(A>B) + 1 - Raw P(B>A)) / 2
   ```

2. **Extreme Preference Selection**: For each prompt, we selected the response pair with the most extreme preference (farthest from 0.5) as the chosen/rejected pair.

## Dataset Statistics

- **Total Samples**: {stats.get('total_samples', 'N/A')}
{split_stats}
## Dataset Structure

Each example contains only the essential columns for SimPO training:
- `prompt`: List with user message in OpenAI chat format
- `chosen`: Full conversation with preferred response in OpenAI chat format
- `rejected`: Full conversation with rejected response in OpenAI chat format

This simplified format is optimized for direct use with SimPO training scripts.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Citation

If you use this dataset, please cite the original preference comparison methodology and any underlying datasets used.
"""
    
    return dataset_card

def main():
    parser = argparse.ArgumentParser(
        description="Convert preference matrices to preference dataset format"
    )
    parser.add_argument("--train_file", type=str, required=True,
                       help="Train split JSON file with preference matrices")
    parser.add_argument("--test_file", type=str, required=True,
                       help="Test split JSON file with preference matrices")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory to save dataset locally")
    parser.add_argument("--repo_id", type=str, required=True,
                       help="HuggingFace repository ID (username/dataset-name)")
    parser.add_argument("--private", action="store_true", default=True,
                       help="Make the repository private")
    parser.add_argument("--dry_run", action="store_true", default=False,
                       help="Create dataset but don't push to hub")
    
    args = parser.parse_args()
    
    # Prepare input files dict
    input_files = {
        'train': args.train_file,
        'test': args.test_file
    }

    # Convert to preference dataset
    dataset_dict = convert_to_preference_dataset(
        input_files,
        args.output_dir,
    )

    # Calculate statistics
    total_samples = sum(len(dataset_dict[split]) for split in dataset_dict.keys())
    split_stats = {split: len(dataset_dict[split]) for split in dataset_dict.keys()}

    stats = {
        'total_samples': total_samples,
        'splits': split_stats
    }

    print(f"\n=== Final Dataset Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    for split, count in split_stats.items():
        print(f"{split.title()} samples: {count}")
    
    if not args.dry_run:
        # Push to hub
        push_to_hub(
            dataset_dict=dataset_dict,
            repo_id=args.repo_id,
            private=args.private
        )
        
        # Create and save dataset card
        card_content = create_dataset_card(args.repo_id, stats)
        
        # Save dataset card locally if output_dir is specified
        if args.output_dir:
            card_path = Path(args.output_dir) / "README.md"
            with open(card_path, 'w') as f:
                f.write(card_content)
            print(f"Dataset card saved to {card_path}")
    else:
        print("Dry run completed. Dataset created but not pushed to hub.")

if __name__ == "__main__":
    main()
