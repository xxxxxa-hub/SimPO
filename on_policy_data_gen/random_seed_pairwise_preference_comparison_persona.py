#!/usr/bin/env python3
"""
Script to generate pairwise preference comparisons with persona-based in-context learning.

Workflow:
1. Filters dataset to a single persona/user
2. Randomly samples k-shot examples from the persona's data as demonstrations
3. Tests on the remaining samples (up to max_samples)
4. For each query, randomizes the order of demonstrations
5. Generates preference matrix comparing two responses (yw vs yl)

The global seed is set once at the beginning for reproducibility, but each query
gets a different random order of demonstrations.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
import json
import os
import argparse
import tqdm
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import logging
import time
import random
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.distributed as dist
from datetime import timedelta
import datetime, torch.distributed as dist
# dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=60))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairwiseComparisonDataset(Dataset):
    """Custom dataset for handling pairwise comparisons with randomized demonstration order per query."""

    def __init__(self, dataset, tokenizer, max_length=None, few_shot_examples=None, query_random_seed=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.few_shot_examples = few_shot_examples or []
        self.query_random_seed = query_random_seed  # Seed for per-query randomization
        self.comparison_data = []

        # Pre-generate all comparison pairs for all samples
        self._prepare_comparisons()

    def _prepare_comparisons(self):
        """Pre-generate all comparison pairs across all samples."""
        for sample_idx, sample in enumerate(self.dataset):
            prompt = sample["prompt"].strip()
            all_responses = [r.strip() for r in sample["all_generated_responses"]]

            # Check we have at least 2 responses
            if len(all_responses) < 2:
                logger.warning(f"Sample {sample_idx} has {len(all_responses)} responses, need at least 2. Skipping.")
                continue

            # Generate all pairwise comparisons for this sample
            for i in range(len(all_responses)):
                for j in range(len(all_responses)):
                    if i == j:
                        continue  # Skip diagonal elements

                    # Store metadata - prompt will be created on-the-fly in __getitem__
                    self.comparison_data.append({
                        'sample_idx': sample_idx,
                        'response_i': i,
                        'response_j': j,
                        'response_a': all_responses[i],
                        'response_b': all_responses[j],
                        'original_prompt': prompt,
                        'all_responses': all_responses,
                        # 'original_rm_scores': sample.get("all_rm_scores", None)
                    })

    def _create_prompt_template(self, question, response_a, response_b, query_idx):
        """Create a prompt template for pairwise comparison with randomized few-shot order.

        Following the approach from preference_datasets.py: no system prompt, no chat template,
        examples and test query in one string.
        """

        # Build the prompt as a single string with few-shot examples
        prompt_text = ""

        # Randomize few-shot examples order for this query
        if self.few_shot_examples:
            # Use query_idx as seed modifier for reproducibility but different order per query
            local_rng = random.Random(self.query_random_seed + query_idx)
            randomized_examples = self.few_shot_examples.copy()
            local_rng.shuffle(randomized_examples)

            for i, example in enumerate(randomized_examples):
                prompt_text += f"# Example {i + 1}\n"
                prompt_text += f"## Question\n{example['question']}\n\n"
                prompt_text += f"[The Start of Assistant A's Answer]\n{example['response_a']}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{example['response_b']}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"
                prompt_text += f"## Preferred answer: [[{example['label']}]]\n\n"

        # Add task header with brief instruction
        prompt_text += "# Task\n"
        if self.few_shot_examples:
            prompt_text += "Given the examples above, evaluate the quality of two AI assistants' responses based on helpfulness, relevance, and accuracy. "
        else:
            prompt_text += "Evaluate the quality of two AI assistants' responses based on helpfulness, relevance, and accuracy. "
        # prompt_text += 'Output your verdict as "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.\n\n'

        # Add the current query
        prompt_text += f"## Question\n{question}\n\n"
        prompt_text += f"[The Start of Assistant A's Answer]\n{response_a}\n"
        prompt_text += f"[The End of Assistant A's Answer]\n\n"
        prompt_text += f"[The Start of Assistant B's Answer]\n{response_b}\n"
        prompt_text += f"[The End of Assistant B's Answer]\n\n"
        prompt_text += "## Preferred answer: [["

        return prompt_text

    def __len__(self):
        return len(self.comparison_data)

    def __getitem__(self, idx):
        item = self.comparison_data[idx]

        # Create prompt with randomized demonstration order
        # Use sample_idx as the query_idx for consistent randomization per query
        prompt_text = self._create_prompt_template(
            item['original_prompt'],
            item['response_a'],
            item['response_b'],
            item['sample_idx']
        )

        return {
            'prompt_text': prompt_text,
            'sample_idx': item['sample_idx'],
            'response_i': item['response_i'],
            'response_j': item['response_j'],
            'response_a': item['response_a'],
            'response_b': item['response_b'],
            'original_prompt': item['original_prompt'],
            'all_responses': item['all_responses'],
            # 'original_rm_scores': item['original_rm_scores']
        }

def make_collate_fn(tokenizer, max_length=None):
    """Create a collate function that uses tokenizer's built-in left padding and precomputed kwargs."""
    # Precompute tokenizer arguments once
    tokenize_kwargs = {
        'return_tensors': "pt",
        'padding': True,  # Dynamic padding to longest in batch
    }
    if max_length is not None:
        tokenize_kwargs.update({
            'truncation': True,
            'max_length': max_length
        })

    def _collate_fn(batch):
        # Extract prompts for tokenizer padding
        prompts = [item['prompt_text'] for item in batch]

        # Use tokenizer's built-in padding (respects padding_side="left")
        tokenized = tokenizer(prompts, **tokenize_kwargs)

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'metadata': [{
                k: v for k, v in item.items()
                if k not in ['prompt_text']
            } for item in batch]
        }

    return _collate_fn


def load_few_shot_examples_from_bt_scores(bt_scores_file, indices):
    """Load few-shot examples from Bradley-Terry scores file.

    Selects the response with highest BT score as chosen and lowest BT score as rejected.
    Note: No shuffling is done here - shuffling happens per-query in the dataset.

    Args:
        bt_scores_file: Path to JSON file with Bradley-Terry scores
        indices: List of sample IDs to use for few-shot examples
    """
    try:
        logger.info(f"Loading few-shot examples from Bradley-Terry scores file: {bt_scores_file}")
        with open(bt_scores_file, 'r') as f:
            bt_data = json.load(f)

        examples = []
        for idx in indices:
            # Access bt_data by list index directly (indices are 0-99, not sample_ids)
            if idx >= len(bt_data):
                logger.warning(f"Index {idx} is out of range for bt_data (size: {len(bt_data)})")
                continue

            sample = bt_data[idx]

            question = sample.get('prompt', '').strip()
            responses = [r.strip() for r in sample.get('responses', [])]
            bt_scores = sample.get('bt_scores', [])
            rankings = sample.get('rankings', [])

            if len(responses) == 0 or len(bt_scores) == 0:
                logger.warning(f"Sample ID {idx} has no responses or scores")
                continue

            # Get best (highest BT score) and worst (lowest BT score) responses
            best_idx = rankings[0]  # First in rankings = highest score
            worst_idx = rankings[-1]  # Last in rankings = lowest score

            chosen = responses[best_idx].strip()
            rejected = responses[worst_idx].strip()

            # Create first example: A=chosen, B=rejected (label=A)
            examples.append({
                'question': question,
                'response_a': chosen,
                'response_b': rejected,
                'label': 'A'
            })

            # Create second example: A=rejected, B=chosen (label=B)
            # Always create both to ensure balanced demonstrations
            examples.append({
                'question': question,
                'response_a': rejected,
                'response_b': chosen,
                'label': 'B'
            })

        # NO shuffling here - will be done per-query
        logger.info(f"Loaded {len(examples)} few-shot examples from Bradley-Terry scores")
        return examples

    except Exception as e:
        logger.warning(f"Failed to load few-shot examples from BT scores: {e}")
        return []

def load_few_shot_examples_from_persona_data(persona_data, indices):
    """Load few-shot examples from persona dataset.

    For persona dataset, yw is the winning/preferred response and yl is the losing response.

    Args:
        persona_data: List of persona dataset samples (converted format)
        indices: List of sample indices to use for few-shot examples

    Returns:
        List of few-shot examples
    """
    try:
        logger.info(f"Loading few-shot examples from persona data")
        examples = []

        for idx in indices:
            if idx >= len(persona_data):
                logger.warning(f"Index {idx} is out of range for persona_data (size: {len(persona_data)})")
                continue

            sample = persona_data[idx]
            question = sample['prompt'].strip()
            responses = sample['all_generated_responses']

            if len(responses) < 2:
                logger.warning(f"Sample {idx} has fewer than 2 responses")
                continue

            # For persona dataset: responses[0] = yw (winner), responses[1] = yl (loser)
            chosen = responses[0].strip()
            rejected = responses[1].strip()

            # Create first example: A=chosen, B=rejected (label=A)
            examples.append({
                'question': question,
                'response_a': chosen,
                'response_b': rejected,
                'label': 'A'
            })

            # Create second example: A=rejected, B=chosen (label=B)
            examples.append({
                'question': question,
                'response_a': rejected,
                'response_b': chosen,
                'label': 'B'
            })

        logger.info(f"Loaded {len(examples)} few-shot examples from persona data")
        return examples

    except Exception as e:
        logger.warning(f"Failed to load few-shot examples from persona data: {e}")
        return []


def filter_dataset_by_persona(dataset, persona_id=None):
    """Filter dataset to only include samples from a specific persona.

    Args:
        dataset: HuggingFace dataset
        persona_id: Specific persona UUID to filter for. If None, uses first persona.

    Returns:
        Filtered dataset and the persona_id used
    """
    # Get all unique persona IDs
    if 'persona_uuid' in dataset.column_names:
        persona_col = 'persona_uuid'
    elif 'score_persona' in dataset.column_names:
        # Extract persona_uuid from score_persona dict
        persona_ids = [sample['score_persona'].get('persona_uuid') if isinstance(sample.get('score_persona'), dict) else None
                      for sample in dataset]
        # If we need to filter by persona_uuid from nested dict
        if persona_id is None:
            # Get first non-None persona
            persona_id = next((pid for pid in persona_ids if pid is not None), None)

        logger.info(f"Filtering for persona: {persona_id}")
        filtered_indices = [i for i, pid in enumerate(persona_ids) if pid == persona_id]
        filtered_dataset = dataset.select(filtered_indices)
        logger.info(f"Filtered dataset size: {len(filtered_dataset)} samples for persona {persona_id}")
        return filtered_dataset, persona_id
    else:
        logger.warning("No persona_uuid column found, returning original dataset")
        return dataset, None

def prepare_persona_dataset_as_pairwise(dataset):
    """Convert persona dataset format to expected pairwise comparison format.

    The persona dataset has 'question', 'yw', 'yl' fields.
    We'll create a format compatible with the existing code by treating yw and yl as two responses.

    Args:
        dataset: Filtered persona dataset

    Returns:
        List of dicts with 'prompt' and 'all_generated_responses' keys
    """
    converted_data = []
    for sample in dataset:
        # Extract question
        prompt = sample['x']

        # Get the two responses (winner and loser)
        yw = sample.get('yw', '').strip()
        yl = sample.get('yl', '').strip()

        # Create pairwise comparison format with 2 responses
        # We'll duplicate to create 5 responses as expected by original code,
        # or modify to work with just 2
        converted_data.append({
            'prompt': prompt,
            'all_generated_responses': [yw, yl],  # Just two responses
            'all_rm_scores': None  # No RM scores for this dataset
        })

    return converted_data

def sample_few_shot_indices(num_annotated=100, k_shot=4, seed=42):
    """Randomly sample k indices from the annotated examples.

    Args:
        num_annotated: Total number of annotated examples available
        k_shot: Number of examples to sample
        seed: Random seed for reproducibility

    Returns:
        List of k randomly sampled indices
    """
    rng = random.Random(seed)
    indices = rng.sample(range(num_annotated), k_shot)
    return sorted(indices)  # Sort for consistency in logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pairwise preference comparisons with randomized few-shot")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to use for preference evaluation")
    parser.add_argument("--dataset_name", type=str, default="sher222/persona-iterative-responses",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--split", type=str, default="train",
                       help="Split of the dataset to process")
    parser.add_argument("--persona_id", type=str, default=None,
                       help="Specific persona UUID to filter for. If None, uses first persona found.")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device for processing comparisons")
    parser.add_argument("--max_samples", type=int, default=512,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--use_compilation", action="store_true", default=False,
                       help="Whether to use torch compilation for speedup")
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                       help="Whether to use Flash Attention 2 if available")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum sequence length for tokenization (None for no truncation)")
    parser.add_argument("--k_shot", type=int, default=4,
                       help="Number of few-shot examples to sample from persona data as demonstrations")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for few-shot sampling and demonstration order (used if --seeds not provided)")
    parser.add_argument("--seeds", type=int, nargs='+', default=None,
                       help="Multiple random seeds to run experiments for (e.g., --seeds 42 123 456)")
    parser.add_argument("--output_dir", type=str, default="./random_seed_results",
                       help="Directory to save results")
    return parser.parse_args()

def get_preference_probabilities_batch_accelerate(model, batch, token_a_id, token_b_id, accelerator):
    """
    Compute preference probabilities for a batch using Accelerate.
    """
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    with torch.no_grad(), torch.autocast(device_type='cuda' if accelerator.device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
        # Get model outputs for the batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        # Get the logits for the last token position for each sequence
        batch_size = logits.shape[0]

        results = []
        for i in range(batch_size):
            last_token_logits = logits[i, -1, :]  # Shape: (vocab_size,)

            # Get logits for tokens "A" and "B"
            logit_a = last_token_logits[token_a_id].item()
            logit_b = last_token_logits[token_b_id].item()

            # Compute probability using softmax
            logits_ab = torch.tensor([logit_a, logit_b])
            probs_ab = F.softmax(logits_ab, dim=0)
            prob_a_over_b = probs_ab[0].item()  # Probability of A

            results.append((prob_a_over_b, logit_a, logit_b))

    return results

def reconstruct_preference_matrices(all_comparison_results):
    """
    Reconstruct preference matrices from distributed comparison results.

    Returns:
        final_results: List of results with preference matrices
        accuracy_metrics: Dict with accuracy statistics
    """
    # Group results by sample_idx
    sample_results = {}
    for result in all_comparison_results:
        sample_idx = result['sample_idx']
        if sample_idx not in sample_results:
            sample_results[sample_idx] = {
                'comparisons': {},
                'metadata': {
                    'original_prompt': result['original_prompt'],
                    'all_responses': result['all_responses'],
                    # 'original_rm_scores': result['original_rm_scores']
                }
            }

        key = f"{result['response_i']}_vs_{result['response_j']}"
        sample_results[sample_idx]['comparisons'][key] = {
            'response_a': result['response_a'],
            'response_b': result['response_b'],
            'prob_a_over_b': result['prob_a_over_b'],
            'logit_a': result['logit_a'],
            'logit_b': result['logit_b'],
        }

    # Convert to final format with preference matrices and calculate accuracy
    final_results = []
    correct_predictions = 0
    total_predictions = 0

    for sample_idx in sorted(sample_results.keys()):
        sample_data = sample_results[sample_idx]
        metadata = sample_data['metadata']
        comparisons = sample_data['comparisons']

        # Reconstruct preference matrix
        n_responses = len(metadata['all_responses'])
        preference_matrix = np.full((n_responses, n_responses), np.nan)

        for key, comparison in comparisons.items():
            i, j = map(int, key.split('_vs_'))
            preference_matrix[i, j] = comparison['prob_a_over_b']

        # Calculate accuracy for this sample
        # For persona dataset: responses[0] = yw (winner), responses[1] = yl (loser)
        # Ground truth: response 0 should be preferred over response 1
        # Check if preference_matrix[0, 1] > 0.5 (model prefers response 0 over 1)
        is_correct = False
        if n_responses >= 2:
            prob_0_over_1 = preference_matrix[0, 1]
            if not np.isnan(prob_0_over_1):
                is_correct = prob_0_over_1 > 0.5
                correct_predictions += int(is_correct)
                total_predictions += 1

        # Convert NaN values to None for JSON serialization
        preference_matrix_list = preference_matrix.tolist()
        preference_matrix_serializable = [
            [None if (isinstance(val, float) and np.isnan(val)) else val for val in row]
            for row in preference_matrix_list
        ]

        result = {
            "sample_id": sample_idx,
            "prompt": metadata['original_prompt'],
            "responses": metadata['all_responses'],
            "preference_matrix": preference_matrix_serializable,
            "detailed_comparisons": comparisons,
            # "original_rm_scores": metadata['original_rm_scores'],
            "is_correct": bool(is_correct),
            "predicted_winner_prob": float(preference_matrix[0, 1]) if not np.isnan(preference_matrix[0, 1]) else None
        }

        final_results.append(result)

    # Calculate accuracy metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    accuracy_metrics = {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions
    }

    return final_results, accuracy_metrics

def run_experiment_for_seed(seed, args, accelerator, model, tokenizer, token_a_id, token_b_id, persona_data, persona_id=None):
    """Run the preference comparison experiment for a single seed.

    Args:
        persona_data: List of persona dataset samples in converted format
        persona_id: The persona UUID being used
    """

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment with seed: {seed}")
        logger.info(f"{'='*80}")

    # Total available samples for this persona
    total_samples = len(persona_data)

    # First, deterministically select test set (independent of seed/k_shot)
    # This ensures the same test examples are used across all experiments
    if args.max_samples and args.max_samples < total_samples:
        test_indices = list(range(args.max_samples))
    else:
        test_indices = list(range(total_samples))

    if accelerator.is_main_process:
        logger.info(f"Test set indices: first {len(test_indices)} samples (0-{len(test_indices)-1})")

    # Create holdout set: all samples NOT in test set
    holdout_indices = [i for i in range(total_samples) if i not in test_indices]

    if accelerator.is_main_process:
        logger.info(f"Holdout set size: {len(holdout_indices)} samples")

    # Now sample k-shot demonstrations from the holdout set using the seed
    if len(holdout_indices) < args.k_shot:
        raise ValueError(f"Not enough holdout samples ({len(holdout_indices)}) to sample {args.k_shot} few-shot examples. "
                        f"Reduce max_samples or k_shot.")

    few_shot_indices = sample_few_shot_indices(
        num_annotated=len(holdout_indices),
        k_shot=args.k_shot,
        seed=seed
    )
    # Map back to original indices in persona_data
    few_shot_indices = [holdout_indices[i] for i in few_shot_indices]

    if accelerator.is_main_process:
        logger.info(f"Sampled few-shot indices from holdout: {few_shot_indices}")

    # Load few-shot examples from persona data
    few_shot_examples = load_few_shot_examples_from_persona_data(
        persona_data=persona_data,
        indices=few_shot_indices
    )

    if accelerator.is_main_process:
        logger.info(f"Loaded {len(few_shot_examples)} few-shot examples")

    test_data = [persona_data[i] for i in test_indices]

    if accelerator.is_main_process:
        logger.info(f"Test set size: {len(test_data)} samples (excluded {len(few_shot_indices)} demonstration samples)")

    # Create custom dataset with randomized demonstration order per query
    comparison_dataset = PairwiseComparisonDataset(
        test_data,  # Use test_data instead of full dataset
        tokenizer,
        args.max_length,
        few_shot_examples,
        query_random_seed=seed  # Use seed for per-query randomization
    )

    # Create collate function with tokenizer and max_length
    collate = make_collate_fn(tokenizer, args.max_length)

    dataloader = DataLoader(
        comparison_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0
    )

    # Prepare dataloader with accelerator
    dataloader = accelerator.prepare(dataloader)

    if accelerator.is_main_process:
        logger.info(f"Total comparisons: {len(comparison_dataset)}")
        logger.info(f"Batches per process: {len(dataloader)}")

    # Process batches
    all_results = []
    start_time = time.time()

    with accelerator.main_process_first():
        progress_bar = tqdm.tqdm(
            dataloader,
            desc=f"Seed {seed} - Processing comparisons",
            disable=not accelerator.is_main_process
        )

    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Get preference probabilities for the batch
            batch_results = get_preference_probabilities_batch_accelerate(
                model, batch, token_a_id, token_b_id, accelerator
            )

            # Combine with metadata
            for i, (prob_a_over_b, logit_a, logit_b) in enumerate(batch_results):
                metadata = batch['metadata'][i]
                result = {
                    'sample_idx': metadata['sample_idx'],
                    'response_i': metadata['response_i'],
                    'response_j': metadata['response_j'],
                    'response_a': metadata['response_a'],
                    'response_b': metadata['response_b'],
                    'original_prompt': metadata['original_prompt'],
                    'all_responses': metadata['all_responses'],
                    # 'original_rm_scores': metadata['original_rm_scores'],
                    'prob_a_over_b': prob_a_over_b,
                    'logit_a': logit_a,
                    'logit_b': logit_b
                }
                all_results.append(result)

            # Log progress periodically
            if accelerator.is_main_process and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (len(dataloader) - batch_idx - 1) / 60
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches | "
                          f"Avg: {avg_time:.2f}s/batch | ETA: {eta:.1f}min")

        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    # Save intermediate results per rank
    rank = accelerator.process_index
    # Include persona_id in filename to avoid overwriting results from different personas
    persona_str = persona_id if persona_id else "unknown"
    part_file = f"{args.output_dir}/seed_{seed}_rank_{rank}_persona_{persona_str}.jsonl"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(part_file, "w") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            

    accelerator.wait_for_everyone()

    # Only main process handles final processing and saving
    if accelerator.is_main_process:
        # Gather results from all ranks
        gathered_results = []
        for r in range(accelerator.num_processes):
            rank_file = f"{args.output_dir}/seed_{seed}_rank_{r}_persona_{persona_str}.jsonl"
            with open(rank_file) as f:
                for line in f:
                    gathered_results.append(json.loads(line))

        logger.info(f"Gathered {len(gathered_results)} comparison results for seed {seed}")

        # Reconstruct preference matrices and calculate accuracy
        final_results, accuracy_metrics = reconstruct_preference_matrices(gathered_results)

        # Log accuracy metrics
        logger.info(f"\n{'='*80}")
        logger.info(f"ACCURACY METRICS")
        logger.info(f"{'='*80}")
        logger.info(f"Accuracy: {accuracy_metrics['accuracy']:.4f} ({accuracy_metrics['accuracy']*100:.2f}%)")
        logger.info(f"Correct predictions: {accuracy_metrics['correct_predictions']}/{accuracy_metrics['total_predictions']}")
        logger.info(f"{'='*80}\n")

        # Save final results for this seed
        output_file = f"{args.output_dir}/pairwise_preferences_seed_{seed}_k{args.k_shot}_persona_{persona_str}.json"
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        # Also save metadata about the experiment
        metadata_file = f"{args.output_dir}/metadata_seed_{seed}_k{args.k_shot}_persona_{persona_str}.json"
        metadata = {
            "seed": seed,
            "k_shot": args.k_shot,
            "few_shot_indices": few_shot_indices,
            "num_test_samples": len(final_results),
            "total_persona_samples": total_samples,
            "persona_id": persona_id,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "accuracy_metrics": accuracy_metrics
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Seed {seed} complete. Processed {len(final_results)} samples with {accuracy_metrics['accuracy']*100:.2f}% accuracy")

        total_time = time.time() - start_time
        logger.info(f"Time for seed {seed}: {total_time / 60:.1f} minutes")

    accelerator.wait_for_everyone()

def main():
    # Initialize accelerator with DDP settings
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Only log on main process
    if accelerator.is_main_process:
        logger.info(f"{'='*80}")
        logger.info(f"Persona-Based In-Context Learning Experiment")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Dataset: {args.dataset_name}")
        logger.info(f"Split: {args.split}")
        logger.info(f"Persona ID: {args.persona_id if args.persona_id else 'auto (first found)'}")
        logger.info(f"K-shot: {args.k_shot}")
        logger.info(f"Max test samples: {args.max_samples}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Output directory: {args.output_dir}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Set pad token if not set and configure for left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # Determine attention implementation
    attn_impl = None
    if args.use_flash_attention and torch.cuda.is_available():
        try:
            attn_impl = "flash_attention_2"
            if accelerator.is_main_process:
                logger.info("Using Flash Attention 2")
        except:
            if accelerator.is_main_process:
                logger.warning("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    model.eval()

    # Pre-compute token IDs for "A" and "B"
    token_a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    token_b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    if accelerator.is_main_process:
        logger.info(f"Token A ID: {token_a_id}, Token B ID: {token_b_id}")

    # Enable torch compilation
    if args.use_compilation:
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            if accelerator.is_main_process:
                logger.info("Model compiled successfully with max-autotune")
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning(f"Model compilation failed: {e}")

    # Prepare model with accelerator
    model = accelerator.prepare(model)

    # Load dataset
    if accelerator.is_main_process:
        logger.info(f"Loading dataset: {args.dataset_name}")

    dataset = load_dataset(args.dataset_name, split=args.split)

    # Filter by persona
    if accelerator.is_main_process:
        logger.info(f"Filtering dataset by persona...")

    filtered_dataset, persona_id = filter_dataset_by_persona(dataset, args.persona_id)

    if accelerator.is_main_process:
        logger.info(f"Using persona: {persona_id}")
        logger.info(f"Persona-filtered dataset size: {len(filtered_dataset)}")

    # Convert persona dataset to expected format
    if accelerator.is_main_process:
        logger.info("Converting persona dataset to pairwise comparison format...")

    persona_data = prepare_persona_dataset_as_pairwise(filtered_dataset)

    if accelerator.is_main_process:
        logger.info(f"Converted {len(persona_data)} samples")

    # Determine which seeds to use
    if args.seeds is not None:
        seeds_to_run = args.seeds
        if accelerator.is_main_process:
            logger.info(f"Running experiments for multiple seeds: {seeds_to_run}")
    else:
        seeds_to_run = [args.seed]
        if accelerator.is_main_process:
            logger.info(f"Running experiment for single seed: {args.seed}")

    # Run experiment for each seed
    # Note: max_samples will be used to limit test set size (after excluding k_shot demonstrations)
    for seed in seeds_to_run:
        run_experiment_for_seed(
            seed=seed,
            args=args,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            token_a_id=token_a_id,
            token_b_id=token_b_id,
            persona_data=persona_data,
            persona_id=persona_id
        )

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"All experiments complete!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
