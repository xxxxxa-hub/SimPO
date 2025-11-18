#!/usr/bin/env python3
"""
Test k-tuples of demonstration contexts with SNR-based selection.

For k demonstrations per context, select n_best best and n_worst worst candidates,
creating P_k^{n_best+n_worst} ordered k-tuples:
1. Randomly assign labels (A/B) to each demonstration in each k-tuple
2. Test on 50 validation examples (averaging over 2 test orderings)
3. Compute gain vs no-context baseline
4. Select top N k-tuples by SNR (mean gain / std gain)
5. Test top N k-tuples on test set with same label assignments

Example: k=4, n_best=4, n_worst=4 gives 8 candidates
         P_4^8 = 8*7*6*5 = 1,680 ordered 4-tuples
         Each tested on 50 validation × 2 orderings = 168,000 total examples
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import numpy as np
import logging
import argparse
import os
from tqdm import tqdm
import random
from itertools import permutations
import json
import pickle
from demo_selection_utils import (
    filter_dataset_by_persona,
    prepare_persona_dataset_as_pairwise,
    compute_best_worst_demo_indices
)

# Set NCCL timeout (in seconds) - default is 600s, increase to 30 minutes
os.environ.setdefault('NCCL_TIMEOUT', '7200')
os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # Enable blocking wait for better error messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt_template(question, response_a, response_b, demo_examples=None, demo_flips=None):
    """
    Create a prompt for pairwise comparison with k demonstration contexts.

    Args:
        question: Test question
        response_a: Response A for test
        response_b: Response B for test
        demo_examples: List of k demonstration examples (each with 'prompt' and 'all_generated_responses')
        demo_flips: List of k boolean flags indicating whether to flip labels for each demo
    """
    prompt_text = ""

    # Add demonstrations if provided
    if demo_examples is not None and demo_flips is not None:
        for i, (demo_example, demo_flip) in enumerate(zip(demo_examples, demo_flips)):
            ctx_question = demo_example['prompt']
            ctx_yw = demo_example['all_generated_responses'][0]
            ctx_yl = demo_example['all_generated_responses'][1]

            prompt_text += f"# Example {i+1}\n"
            prompt_text += f"## Question\n{ctx_question}\n\n"

            if demo_flip:
                prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yl}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yw}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"
                prompt_text += "## Preferred answer: [[B]]\n\n"
            else:
                prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yw}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yl}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"
                prompt_text += "## Preferred answer: [[A]]\n\n"

    # Add task header
    prompt_text += "# Task\n"
    if demo_examples is not None and len(demo_examples) > 0:
        prompt_text += "Given the examples above, evaluate the quality of two AI assistants' responses.\n\n"
    else:
        prompt_text += "Evaluate the quality of two AI assistants' responses.\n\n"

    # Add the current query
    prompt_text += f"## Question\n{question}\n\n"
    prompt_text += f"[The Start of Assistant A's Answer]\n{response_a}\n"
    prompt_text += f"[The End of Assistant A's Answer]\n\n"
    prompt_text += f"[The Start of Assistant B's Answer]\n{response_b}\n"
    prompt_text += f"[The End of Assistant B's Answer]\n\n"
    prompt_text += "## Preferred answer: [["

    return prompt_text


class KTupleValidationDataset(Dataset):
    """Dataset for all (k-tuple, validation, ordering) combinations."""

    def __init__(self, all_tuples, candidate_examples, tuple_labels, validation_set):
        """
        Args:
            all_tuples: List of k-tuples, where each k-tuple is a tuple of k indices
            candidate_examples: List of candidate demonstration examples
            tuple_labels: List of k-tuples of boolean flags for label flipping
            validation_set: Validation examples
        """
        self.all_tuples = all_tuples
        self.candidate_examples = candidate_examples
        self.tuple_labels = tuple_labels
        self.validation_set = validation_set

        # Create all (tuple_idx, val_idx, ordering) combinations
        self.items = []
        for tuple_idx in range(len(all_tuples)):
            for val_idx in range(len(validation_set)):
                # Two orderings per (tuple, validation) combination
                self.items.append((tuple_idx, val_idx, 0))  # ordering 0: A=yw, B=yl
                self.items.append((tuple_idx, val_idx, 1))  # ordering 1: A=yl, B=yw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tuple_idx, val_idx, ordering = self.items[idx]

        k_tuple = self.all_tuples[tuple_idx]
        k_tuple_labels = self.tuple_labels[tuple_idx]

        val_example = self.validation_set[val_idx]

        # Prepare demonstration data
        demo_data = []
        for demo_idx, demo_flip in zip(k_tuple, k_tuple_labels):
            demo_example = self.candidate_examples[demo_idx]
            demo_data.append({
                'prompt': demo_example['prompt'],
                'yw': demo_example['all_generated_responses'][0],
                'yl': demo_example['all_generated_responses'][1],
                'flip': demo_flip
            })

        return {
            'tuple_idx': tuple_idx,
            'val_idx': val_idx,
            'ordering': ordering,
            'demo_data': demo_data,
            'val_prompt': val_example['prompt'],
            'val_yw': val_example['all_generated_responses'][0],
            'val_yl': val_example['all_generated_responses'][1],
        }


def collate_fn(batch):
    """Collate function."""
    return batch


def compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch):
    """
    Compute log probabilities for a batch of (k-tuple, validation, ordering) examples.

    Returns: List of dicts with tuple_idx, val_idx, ordering, and log_prob
    """
    all_prompts = []
    metadata = []

    for item in batch:
        demo_examples = []
        demo_flips = []
        for demo in item['demo_data']:
            demo_examples.append({
                'prompt': demo['prompt'],
                'all_generated_responses': [demo['yw'], demo['yl']]
            })
            demo_flips.append(demo['flip'])

        val_prompt = item['val_prompt']
        val_yw = item['val_yw']
        val_yl = item['val_yl']
        ordering = item['ordering']

        # Create prompt based on ordering
        if ordering == 0:
            # A=yw, B=yl -> correct is A
            prompt = create_prompt_template(
                val_prompt, val_yw, val_yl,
                demo_examples=demo_examples,
                demo_flips=demo_flips)
            correct_token = 0  # A
        else:
            # A=yl, B=yw -> correct is B
            prompt = create_prompt_template(
                val_prompt, val_yl, val_yw,
                demo_examples=demo_examples,
                demo_flips=demo_flips)
            correct_token = 1  # B

        all_prompts.append(prompt)
        metadata.append({
            'tuple_idx': item['tuple_idx'],
            'val_idx': item['val_idx'],
            'ordering': ordering,
            'correct_token': correct_token
        })

    # Tokenize all prompts at once
    inputs = tokenizer(all_prompts, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Get last token logits
        last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # Get logits for tokens A and B
        logit_a = last_token_logits[:, token_a_id]
        logit_b = last_token_logits[:, token_b_id]

        # Compute log probabilities
        logits_ab = torch.stack([logit_a, logit_b], dim=1)  # (batch_size, 2)
        log_probs_ab = F.log_softmax(logits_ab, dim=1)  # (batch_size, 2)

    # Extract results
    results = []
    for i, meta in enumerate(metadata):
        log_prob_correct = log_probs_ab[i, meta['correct_token']].item()

        results.append({
            'tuple_idx': meta['tuple_idx'],
            'val_idx': meta['val_idx'],
            'ordering': meta['ordering'],
            'log_prob': log_prob_correct
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test k-tuple contexts with SNR selection")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to use for evaluation")
    parser.add_argument("--dataset_name", type=str,
                       default="sher222/persona-iterative-responses",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--persona_id", type=str,
                       default="cdf7cefb-7341-41d7-a193-ff0f2f962cf9",
                       help="Specific persona UUID to filter for")
    parser.add_argument("--icl_gain_results_file", type=str,
                       required=True,
                       help="Path to .npy file with ICL gain results")
    parser.add_argument("--output_dir", type=str,
                       default="./icl_snr_results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for label assignment")
    parser.add_argument("--k", type=int, default=4,
                       help="Number of demonstrations per context (k)")
    parser.add_argument("--n_best", type=int, default=4,
                       help="Number of best demonstrations to select as candidates")
    parser.add_argument("--n_worst", type=int, default=4,
                       help="Number of worst demonstrations to select as candidates")
    parser.add_argument("--top_n", type=int, default=5,
                       help="Number of top k-tuples to select and test")
    parser.add_argument("--rank_by", type=str, default="snr_prob_gain",
                       choices=["snr_prob_gain", "mean_prob_gain", "snr_acc_gain", "mean_acc_gain", "snr_prob_gain_filtered"],
                       help="Metric to rank demonstrations by")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize accelerator with extended timeout for large data transfers
    from datetime import timedelta
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # 30 minutes timeout
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    if accelerator.is_main_process:
        logger.info(f"Testing {args.k}-tuples of demonstration contexts with SNR selection")
        logger.info(f"Loading model: {args.model_name}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Batch size per device: {args.batch_size}")
        logger.info(f"k (demonstrations per context): {args.k}")
        logger.info(f"n_best: {args.n_best}, n_worst: {args.n_worst}")
        logger.info(f"Total candidates: {args.n_best + args.n_worst}")
        logger.info(f"Random seed: {args.seed}")

    # Load ICL gain results on all processes for consistency
    icl_gain_results = np.load(args.icl_gain_results_file)

    if accelerator.is_main_process:
        logger.info(f"\nICL gain results shape: {icl_gain_results.shape}")

        # Compute best and worst demonstration indices
        logger.info(f"Computing top {args.n_best} best and {args.n_worst} worst demonstrations using {args.rank_by}...")
        best_demo_indices, worst_demo_indices, metrics = compute_best_worst_demo_indices(
            icl_gain_results, top_k=max(args.n_best, args.n_worst), rank_by=args.rank_by)

        # Take only the requested number
        best_demo_indices = best_demo_indices[:args.n_best]
        worst_demo_indices = worst_demo_indices[:args.n_worst]

        # Combine to get candidate demonstrations
        candidate_demo_indices = best_demo_indices + worst_demo_indices
        num_candidates = len(candidate_demo_indices)

        logger.info(f"\nBest {args.n_best} demonstrations: {best_demo_indices}")
        logger.info(f"Worst {args.n_worst} demonstrations: {worst_demo_indices}")
        logger.info(f"Number of candidate demonstrations: {num_candidates}")

        # Calculate number of k-tuples: P_k^n = n!/(n-k)!
        import math
        num_tuples = math.perm(num_candidates, args.k)
        logger.info(f"Number of {args.k}-tuples: P_{args.k}^{num_candidates} = {num_tuples}")

    # Broadcast to all processes
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            broadcast_data = {
                'candidate_demo_indices': candidate_demo_indices,
                'num_candidates': num_candidates,
                'best_demo_indices': best_demo_indices,
                'worst_demo_indices': worst_demo_indices,
                'num_tuples': num_tuples
            }
        else:
            broadcast_data = None

        import torch.distributed as dist
        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        if not accelerator.is_main_process:
            candidate_demo_indices = broadcast_list[0]['candidate_demo_indices']
            num_candidates = broadcast_list[0]['num_candidates']
            best_demo_indices = broadcast_list[0]['best_demo_indices']
            worst_demo_indices = broadcast_list[0]['worst_demo_indices']
            num_tuples = broadcast_list[0]['num_tuples']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Get token IDs
    token_a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    token_b_id = tokenizer.encode("B", add_special_tokens=False)[0]

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train")
    filtered_dataset, actual_persona_id = filter_dataset_by_persona(dataset, args.persona_id)
    persona_data = prepare_persona_dataset_as_pairwise(filtered_dataset)

    # Split the data
    test_set = persona_data[:512]
    validation_set = persona_data[512:562]
    training_examples = persona_data[562:]

    if accelerator.is_main_process:
        logger.info(f"\nTest set: {len(test_set)} examples")
        logger.info(f"Validation set: {len(validation_set)} examples")
        logger.info(f"Training examples: {len(training_examples)} examples")

    # Select candidate examples using the computed indices
    candidate_examples = [training_examples[idx] for idx in candidate_demo_indices]

    # Generate all ordered k-tuples
    all_tuples = list(permutations(range(num_candidates), args.k))
    num_tuples = len(all_tuples)

    # For each k-tuple, randomly assign labels to each demonstration
    # Explicitly ensure balanced labels: exactly k//2 True and k//2 False
    tuple_labels = []
    for _ in range(num_tuples):
        # Create a balanced list with exactly k//2 True and k//2 False values
        num_true = args.k // 2
        num_false = args.k - num_true
        labels_list = [True] * num_true + [False] * num_false
        random.shuffle(labels_list)
        labels = tuple(labels_list)
        tuple_labels.append(labels)

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 1: Testing {num_tuples} {args.k}-tuples on validation set")
        logger.info(f"{'='*80}")

    # Create dataset
    tuple_val_dataset = KTupleValidationDataset(all_tuples, candidate_examples, tuple_labels, validation_set)

    if accelerator.is_main_process:
        logger.info(f"Total examples in dataset: {len(tuple_val_dataset)} ({num_tuples} tuples × {len(validation_set)} val × 2 orderings)")

    # Create dataloader
    dataloader = DataLoader(
        tuple_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Prepare with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    # Process batches
    all_results = []

    if accelerator.is_main_process:
        logger.info("Starting computation on validation set...")

    for batch in tqdm(dataloader, desc="Processing", disable=not accelerator.is_main_process):
        batch_results = compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch)
        all_results.extend(batch_results)

    # Save each rank's results to a separate file instead of gathering
    os.makedirs(args.output_dir, exist_ok=True)
    rank_file = os.path.join(args.output_dir, f"validation_results_rank{accelerator.process_index}.pkl")
    with open(rank_file, 'wb') as f:
        pickle.dump(all_results, f)

    logger.info(f"Rank {accelerator.process_index}: Saved {len(all_results)} results to {rank_file}")

    # Wait for all processes to finish saving
    accelerator.wait_for_everyone()

    # Only main process loads and combines all results
    if accelerator.is_main_process:
        logger.info("Loading results from all ranks...")
        all_results = []
        for rank_idx in range(accelerator.num_processes):
            rank_file = os.path.join(args.output_dir, f"validation_results_rank{rank_idx}.pkl")
            with open(rank_file, 'rb') as f:
                rank_results = pickle.load(f)
                all_results.extend(rank_results)
            # # Clean up rank file after loading
            # os.remove(rank_file)

        logger.info(f"Loaded {len(all_results)} results from {accelerator.num_processes} ranks")

        # Reconstruct result array: (num_tuples, num_val, 2 orderings)
        log_probs_raw = np.zeros((num_tuples, len(validation_set), 2))

        for result in all_results:
            tuple_idx = result['tuple_idx']
            val_idx = result['val_idx']
            ordering = result['ordering']
            log_probs_raw[tuple_idx, val_idx, ordering] = result['log_prob']

        # Average probabilities across orderings, then convert back to log prob
        probs_raw = np.exp(log_probs_raw)
        avg_probs = probs_raw.mean(axis=2)  # (num_tuples, num_val)
        log_probs_with_context = np.log(avg_probs + 1e-10)

        logger.info(f"Log probs with context shape: {log_probs_with_context.shape}")

        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 2: Computing no-context baseline on validation set")
        logger.info(f"{'='*80}")

        # Compute no-context baseline
        baseline_results = []
        for val_idx in range(len(validation_set)):
            val_example = validation_set[val_idx]
            val_yw = val_example['all_generated_responses'][0]
            val_yl = val_example['all_generated_responses'][1]
            val_prompt = val_example['prompt']

            # Two orderings
            prompts = [
                create_prompt_template(val_prompt, val_yw, val_yl, None, None),
                create_prompt_template(val_prompt, val_yl, val_yw, None, None)
            ]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                logit_a = logits[:, token_a_id]
                logit_b = logits[:, token_b_id]
                logits_ab = torch.stack([logit_a, logit_b], dim=1)
                log_probs_ab = F.log_softmax(logits_ab, dim=1)

                log_prob_1 = log_probs_ab[0, 0].item()  # Correct is A
                log_prob_2 = log_probs_ab[1, 1].item()  # Correct is B

            # Average probabilities
            prob_1 = np.exp(log_prob_1)
            prob_2 = np.exp(log_prob_2)
            avg_prob = (prob_1 + prob_2) / 2.0
            baseline_results.append(np.log(avg_prob + 1e-10))

        log_probs_no_context = np.array(baseline_results)
        logger.info(f"No-context baseline computed: shape {log_probs_no_context.shape}")
        logger.info(f"Mean log prob: {log_probs_no_context.mean():.4f}")

        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 3: Computing gain and SNR")
        logger.info(f"{'='*80}")

        # Compute gain
        gain = log_probs_with_context - log_probs_no_context[np.newaxis, :]  # (num_tuples, 50)

        # For each k-tuple, compute mean and std of gain
        mean_gain = gain.mean(axis=1)  # (num_tuples,)
        std_gain = gain.std(axis=1)    # (num_tuples,)

        # Compute SNR
        snr = mean_gain / (std_gain + 1e-10)  # (num_tuples,)

        # Select top N k-tuples by SNR
        top_n_indices = np.argsort(snr)[-args.top_n:][::-1]

        logger.info(f"\nTop {args.top_n} {args.k}-tuples by SNR:")
        for rank, tuple_idx in enumerate(top_n_indices):
            k_tuple = all_tuples[tuple_idx]
            k_tuple_labels = tuple_labels[tuple_idx]
            logger.info(f"  Rank {rank+1}: Tuple {tuple_idx}")
            logger.info(f"    Demos: {k_tuple}")
            logger.info(f"    Train indices: {[candidate_demo_indices[i] for i in k_tuple]}")
            logger.info(f"    Labels (flip): {k_tuple_labels}")
            logger.info(f"    SNR: {snr[tuple_idx]:.4f}, Mean gain: {mean_gain[tuple_idx]:.4f}, Std gain: {std_gain[tuple_idx]:.4f}")

        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 4: Testing top {args.top_n} {args.k}-tuples on test set")
        logger.info(f"{'='*80}")

        # Create test dataset for top-n k-tuples only
        top_n_tuples = [all_tuples[idx] for idx in top_n_indices]
        top_n_labels = [tuple_labels[idx] for idx in top_n_indices]

        test_dataset = KTupleValidationDataset(top_n_tuples, candidate_examples, top_n_labels, test_set)

        logger.info(f"Test set examples: {len(test_dataset)} ({args.top_n} tuples × {len(test_set)} test × 2 orderings)")

    # Create test dataloader (need to recreate on all processes)
    if accelerator.is_main_process:
        # Broadcast test info
        test_info = {
            'top_n_tuples': top_n_tuples,
            'top_n_labels': top_n_labels
        }
    else:
        test_info = None

    if accelerator.num_processes > 1:
        test_info_list = [test_info]
        torch.distributed.broadcast_object_list(test_info_list, src=0)
        if not accelerator.is_main_process:
            top_n_tuples = test_info_list[0]['top_n_tuples']
            top_n_labels = test_info_list[0]['top_n_labels']

    test_dataset = KTupleValidationDataset(top_n_tuples, candidate_examples, top_n_labels, test_set)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    test_dataloader = accelerator.prepare(test_dataloader)

    # Process test batches
    test_results_all = []

    for batch in tqdm(test_dataloader, desc="Testing on test set", disable=not accelerator.is_main_process):
        batch_results = compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch)
        test_results_all.extend(batch_results)

    # Gather results
    if accelerator.num_processes > 1:
        gathered_test_results = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered_test_results, test_results_all)
        if accelerator.is_main_process:
            test_results_all = []
            for results_list in gathered_test_results:
                test_results_all.extend(results_list)

    if accelerator.is_main_process:
        # Reconstruct test results
        log_probs_test_raw = np.zeros((args.top_n, len(test_set), 2))

        for result in test_results_all:
            tuple_idx = result['tuple_idx']  # This is now in range [0, top_n-1]
            val_idx = result['val_idx']
            ordering = result['ordering']
            log_probs_test_raw[tuple_idx, val_idx, ordering] = result['log_prob']

        # Average across orderings
        probs_test_raw = np.exp(log_probs_test_raw)
        avg_probs_test = probs_test_raw.mean(axis=2)
        log_probs_test_with_context = np.log(avg_probs_test + 1e-10)

        # Compute no-context baseline on test set
        logger.info("Computing no-context baseline on test set...")
        test_baseline_results = []
        for val_idx in range(len(test_set)):
            val_example = test_set[val_idx]
            val_yw = val_example['all_generated_responses'][0]
            val_yl = val_example['all_generated_responses'][1]
            val_prompt = val_example['prompt']

            prompts = [
                create_prompt_template(val_prompt, val_yw, val_yl, None, None),
                create_prompt_template(val_prompt, val_yl, val_yw, None, None)
            ]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                logit_a = logits[:, token_a_id]
                logit_b = logits[:, token_b_id]
                logits_ab = torch.stack([logit_a, logit_b], dim=1)
                log_probs_ab = F.log_softmax(logits_ab, dim=1)

                log_prob_1 = log_probs_ab[0, 0].item()
                log_prob_2 = log_probs_ab[1, 1].item()

            prob_1 = np.exp(log_prob_1)
            prob_2 = np.exp(log_prob_2)
            avg_prob = (prob_1 + prob_2) / 2.0
            test_baseline_results.append(np.log(avg_prob + 1e-10))

        log_probs_test_no_context = np.array(test_baseline_results)

        logger.info(f"Test set no-context baseline: {log_probs_test_no_context.mean():.4f}")

        # Compute test set gain
        test_gain = log_probs_test_with_context - log_probs_test_no_context[np.newaxis, :]

        logger.info(f"\nTest set results:")
        for rank, tuple_idx in enumerate(top_n_indices):
            k_tuple = all_tuples[tuple_idx]
            k_tuple_labels = tuple_labels[tuple_idx]
            mean_test_gain = test_gain[rank, :].mean()
            std_test_gain = test_gain[rank, :].std()
            test_snr = mean_test_gain / (std_test_gain + 1e-10)

            logger.info(f"  Rank {rank+1}: Tuple {tuple_idx}")
            logger.info(f"    Demos: {k_tuple}")
            logger.info(f"    Train indices: {[candidate_demo_indices[i] for i in k_tuple]}")
            logger.info(f"    Labels (flip): {k_tuple_labels}")
            logger.info(f"    Test SNR: {test_snr:.4f}, Mean gain: {mean_test_gain:.4f}, Std gain: {std_test_gain:.4f}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"icl_snr_k{args.k}_results_{actual_persona_id}.npz")

        np.savez(output_file,
                 log_probs_no_context=log_probs_no_context,
                 log_probs_with_context=log_probs_with_context,
                 gain=gain,
                 mean_gain=mean_gain,
                 std_gain=std_gain,
                 snr=snr,
                 top_n_indices=top_n_indices,
                 all_tuples=all_tuples,
                 tuple_labels=tuple_labels,
                 log_probs_test_no_context=log_probs_test_no_context,
                 log_probs_test_with_context=log_probs_test_with_context,
                 test_gain=test_gain,
                 candidate_demo_indices=candidate_demo_indices,
                 best_demo_indices=best_demo_indices,
                 worst_demo_indices=worst_demo_indices,
                 k=args.k,
                 n_best=args.n_best,
                 n_worst=args.n_worst,
                 rank_by=args.rank_by)

        logger.info(f"\nResults saved to: {output_file}")

        # Save all_results as JSON for detailed inspection
        all_results_file = os.path.join(args.output_dir, f"icl_snr_k{args.k}_all_results_{actual_persona_id}.json")
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Detailed results saved to: {all_results_file}")
        logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
