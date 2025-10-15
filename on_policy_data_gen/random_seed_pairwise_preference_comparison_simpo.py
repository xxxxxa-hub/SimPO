#!/usr/bin/env python3
"""
Script to generate pairwise preference comparisons with randomized few-shot examples.

For each of 5 different seeds:
1. Randomly samples k-shot examples from the 100 annotated examples
2. For each query in the dataset, randomizes the order of demonstrations
3. Generates preference matrix on the full train split

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

            # Ensure we have exactly 5 responses
            if len(all_responses) != 5:
                logger.warning(f"Sample {sample_idx} has {len(all_responses)} responses, expected 5. Skipping.")
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
                        'original_rm_scores': sample.get("all_rm_scores", None)
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
                prompt_text += f"## Question\n{example['question']}\n"
                # Determine which response is preferred based on label
                if example['label'] == 'A':
                    preferred = example['response_a']
                    dispreferred = example['response_b']
                else:
                    preferred = example['response_b']
                    dispreferred = example['response_a']
                prompt_text += f"## Preferred Response:\n{preferred}\n"
                prompt_text += f"## Dispreferred Response:\n{dispreferred}\n\n"

        # Add task header with brief instruction
        prompt_text += "# Task\n"
        if self.few_shot_examples:
            prompt_text += "Given the examples above, generate a preferred response to the following question.\n"
        else:
            prompt_text += "Generate a response to the following question.\n"

        # Add the current query
        prompt_text += f"## Question\n{question}\n"
        prompt_text += "## Preferred Response:\n"

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
            'original_rm_scores': item['original_rm_scores']
        }

def make_collate_fn(tokenizer, max_length=None):
    """Create a collate function that tokenizes prompt + responses for log-prob calculation."""
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
        # For each item, we need to tokenize:
        # 1. Prompt + Response A
        # 2. Prompt + Response B

        prompts_with_response_a = []
        prompts_with_response_b = []

        for item in batch:
            prompt = item['prompt_text']
            response_a = item['response_a']
            response_b = item['response_b']

            # Concatenate prompt with each response (no brackets needed)
            prompts_with_response_a.append(prompt + response_a)
            prompts_with_response_b.append(prompt + response_b)

        # Tokenize both sets
        tokenized_a = tokenizer(prompts_with_response_a, **tokenize_kwargs)
        tokenized_b = tokenizer(prompts_with_response_b, **tokenize_kwargs)

        # Also tokenize just the prompt to know where responses start
        prompts_only = [item['prompt_text'] for item in batch]
        tokenized_prompts = tokenizer(prompts_only, **tokenize_kwargs)

        return {
            'input_ids_a': tokenized_a['input_ids'],
            'attention_mask_a': tokenized_a['attention_mask'],
            'input_ids_b': tokenized_b['input_ids'],
            'attention_mask_b': tokenized_b['attention_mask'],
            'prompt_input_ids': tokenized_prompts['input_ids'],
            'prompt_attention_mask': tokenized_prompts['attention_mask'],
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

        # NO shuffling here - will be done per-query
        logger.info(f"Loaded {len(examples)} few-shot examples from Bradley-Terry scores")
        return examples

    except Exception as e:
        logger.warning(f"Failed to load few-shot examples from BT scores: {e}")
        return []


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
    parser.add_argument("--dataset_name", type=str, default="Alligator123/gemma2-ultrafeedback-armorm-false_qa",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--split", type=str, default="train",
                       help="Split of the dataset to process")
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
    parser.add_argument("--bt_scores_file", type=str, default="gpt4_pairwise_preferences_train_100shot_cot_20trials_bt_scores.json",
                       help="Path to Bradley-Terry scores JSON file for few-shot examples")
    parser.add_argument("--num_annotated", type=int, default=100,
                       help="Total number of annotated examples available")
    parser.add_argument("--k_shot", type=int, default=4,
                       help="Number of few-shot examples to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for few-shot sampling and demonstration order")
    parser.add_argument("--output_dir", type=str, default="./random_seed_results",
                       help="Directory to save results")
    return parser.parse_args()

def compute_response_logprob(model, input_ids, attention_mask, prompt_length):
    """
    Compute average log-probability for tokens in the response part only.

    Args:
        model: The language model
        input_ids: Token IDs for prompt + response (shape: [batch_size, seq_len])
        attention_mask: Attention mask (shape: [batch_size, seq_len])
        prompt_length: Number of tokens in the prompt (to know where response starts)

    Returns:
        avg_logprob: Average log-probability per token in the response
        response_length: Number of response tokens
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        # logits[:, :-1] predicts input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather the log probabilities of the actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # Shape: (batch_size, seq_len-1)

        # Only consider tokens in the response (after the prompt)
        # Note: shift means we're looking at positions [prompt_length-1:] in token_log_probs
        # to correspond to tokens [prompt_length:] in the original sequence
        response_token_log_probs = token_log_probs[:, prompt_length-1:]

        # Create mask for valid response tokens (non-padding)
        response_mask = attention_mask[:, prompt_length:].float()

        # Compute average log-prob for response tokens only
        response_log_probs_sum = (response_token_log_probs * response_mask).sum(dim=1)
        response_lengths = response_mask.sum(dim=1)

        # Avoid division by zero
        avg_log_probs = response_log_probs_sum / response_lengths.clamp(min=1)

        return avg_log_probs, response_lengths


def get_preference_probabilities_batch_accelerate(model, batch, tokenizer, accelerator):
    """
    Compute preference probabilities using length-normalized log-probabilities.

    This computes the average log-probability for each response conditioned on the prompt,
    then uses softmax to convert to preference probabilities.
    """
    input_ids_a = batch['input_ids_a']
    attention_mask_a = batch['attention_mask_a']
    input_ids_b = batch['input_ids_b']
    attention_mask_b = batch['attention_mask_b']
    prompt_input_ids = batch['prompt_input_ids']

    with torch.no_grad(), torch.autocast(device_type='cuda' if accelerator.device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
        batch_size = input_ids_a.shape[0]

        # Get prompt length (number of non-padding tokens in prompt)
        prompt_lengths = (prompt_input_ids != tokenizer.eos_token_id).sum(dim=1)

        results = []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i].item()

            # Compute log-probs for response A
            avg_logprob_a, len_a = compute_response_logprob(
                model,
                input_ids_a[i:i+1],
                attention_mask_a[i:i+1],
                prompt_len
            )

            # Compute log-probs for response B
            avg_logprob_b, len_b = compute_response_logprob(
                model,
                input_ids_b[i:i+1],
                attention_mask_b[i:i+1],
                prompt_len
            )

            # Convert to scalar
            avg_logprob_a = avg_logprob_a.item()
            avg_logprob_b = avg_logprob_b.item()

            # Length-normalized rewards (negative average log-prob divided by length)
            # Higher log-prob = better, so we use it directly as reward
            reward_a = avg_logprob_a
            reward_b = avg_logprob_b

            # Use softmax to get preference probability
            rewards = torch.tensor([reward_a, reward_b], dtype=torch.float32)
            probs = F.softmax(rewards, dim=0)
            prob_a_over_b = probs[0].item()

            results.append((prob_a_over_b, reward_a, reward_b))

    return results

def reconstruct_preference_matrices(all_comparison_results):
    """
    Reconstruct preference matrices from distributed comparison results.
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
                    'original_rm_scores': result['original_rm_scores']
                }
            }

        key = f"{result['response_i']}_vs_{result['response_j']}"
        sample_results[sample_idx]['comparisons'][key] = {
            'response_a': result['response_a'],
            'response_b': result['response_b'],
            'prob_a_over_b': result['prob_a_over_b'],
            'reward_a': result['reward_a'],
            'reward_b': result['reward_b'],
        }

    # Convert to final format with preference matrices
    final_results = []
    for sample_idx in sorted(sample_results.keys()):
        sample_data = sample_results[sample_idx]
        metadata = sample_data['metadata']
        comparisons = sample_data['comparisons']

        # Reconstruct 5x5 preference matrix
        n_responses = len(metadata['all_responses'])
        preference_matrix = np.full((n_responses, n_responses), np.nan)

        for key, comparison in comparisons.items():
            i, j = map(int, key.split('_vs_'))
            preference_matrix[i, j] = comparison['prob_a_over_b']

        result = {
            "sample_id": sample_idx,
            "prompt": metadata['original_prompt'],
            "responses": metadata['all_responses'],
            "preference_matrix": preference_matrix.tolist(),
            "detailed_comparisons": comparisons,
            "original_rm_scores": metadata['original_rm_scores']
        }

        final_results.append(result)

    return final_results

def run_experiment_for_seed(seed, args, accelerator, model, tokenizer, dataset):
    """Run the preference comparison experiment for a single seed."""

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment with seed: {seed}")
        logger.info(f"{'='*80}")

    # Sample k-shot indices for this seed
    few_shot_indices = sample_few_shot_indices(
        num_annotated=args.num_annotated,
        k_shot=args.k_shot,
        seed=seed
    )

    if accelerator.is_main_process:
        logger.info(f"Sampled few-shot indices: {few_shot_indices}")

    # Load few-shot examples using the sampled indices
    few_shot_examples = load_few_shot_examples_from_bt_scores(
        bt_scores_file=args.bt_scores_file,
        indices=few_shot_indices
    )

    if accelerator.is_main_process:
        logger.info(f"Loaded {len(few_shot_examples)} few-shot examples")

    # Create custom dataset with randomized demonstration order per query
    comparison_dataset = PairwiseComparisonDataset(
        dataset,
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
                model, batch, tokenizer, accelerator
            )

            # Combine with metadata
            for i, (prob_a_over_b, reward_a, reward_b) in enumerate(batch_results):
                metadata = batch['metadata'][i]
                result = {
                    'sample_idx': metadata['sample_idx'],
                    'response_i': metadata['response_i'],
                    'response_j': metadata['response_j'],
                    'response_a': metadata['response_a'],
                    'response_b': metadata['response_b'],
                    'original_prompt': metadata['original_prompt'],
                    'all_responses': metadata['all_responses'],
                    'original_rm_scores': metadata['original_rm_scores'],
                    'prob_a_over_b': prob_a_over_b,
                    'reward_a': reward_a,
                    'reward_b': reward_b
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
    part_file = f"{args.output_dir}/seed_{seed}_rank_{rank}_new_prompt_4.jsonl"
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
            rank_file = f"{args.output_dir}/seed_{seed}_rank_{r}_new_prompt_4.jsonl"
            with open(rank_file) as f:
                for line in f:
                    gathered_results.append(json.loads(line))

        logger.info(f"Gathered {len(gathered_results)} comparison results for seed {seed}")

        # Reconstruct preference matrices
        final_results = reconstruct_preference_matrices(gathered_results)

        # Save final results for this seed
        output_file = f"{args.output_dir}/pairwise_preferences_seed_{seed}_k{args.k_shot}_new_prompt_4.json"
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        # Also save metadata about the experiment
        metadata_file = f"{args.output_dir}/metadata_seed_{seed}_k{args.k_shot}_new_prompt_4.json"
        metadata = {
            "seed": seed,
            "k_shot": args.k_shot,
            "few_shot_indices": few_shot_indices,
            "num_samples": len(final_results),
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "split": args.split
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Seed {seed} complete. Processed {len(final_results)} samples")

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
        logger.info(f"Random Seed Pairwise Preference Comparison Experiment")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Dataset: {args.dataset_name}")
        logger.info(f"Split: {args.split}")
        logger.info(f"K-shot: {args.k_shot}")
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

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if accelerator.is_main_process:
        logger.info(f"Processing {len(dataset)} samples")

    # Run experiment for the single seed
    run_experiment_for_seed(
        seed=args.seed,
        args=args,
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset
    )

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Experiment complete!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
