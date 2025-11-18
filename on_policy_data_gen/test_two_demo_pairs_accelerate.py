#!/usr/bin/env python3
"""
Test pairs of demonstration contexts with SNR-based selection.

For 90 pairs of contexts (10*9 ordered pairs from 10 candidate demonstrations):
1. Randomly assign balanced labels (A/B) to each demonstration in each pair
2. Test on 50 validation examples (averaging over 2 test orderings)
3. Compute gain vs no-context baseline
4. Select top 5 pairs by SNR (mean gain / std gain)
5. Test top 5 pairs on test set with same label assignments
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
from demo_selection_utils import (
    filter_dataset_by_persona,
    prepare_persona_dataset_as_pairwise,
    compute_best_worst_demo_indices
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt_template(question, response_a, response_b,
                           demo1_example=None, demo1_flip=False,
                           demo2_example=None, demo2_flip=False):
    """Create a prompt for pairwise comparison with two demonstration contexts."""
    prompt_text = ""

    # Add first demo if provided
    if demo1_example is not None:
        ctx_question = demo1_example['prompt']
        ctx_yw = demo1_example['all_generated_responses'][0]
        ctx_yl = demo1_example['all_generated_responses'][1]

        prompt_text += "# Example 1\n"
        prompt_text += f"## Question\n{ctx_question}\n\n"

        if demo1_flip:
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

    # Add second demo if provided
    if demo2_example is not None:
        ctx_question = demo2_example['prompt']
        ctx_yw = demo2_example['all_generated_responses'][0]
        ctx_yl = demo2_example['all_generated_responses'][1]

        prompt_text += "# Example 2\n"
        prompt_text += f"## Question\n{ctx_question}\n\n"

        if demo2_flip:
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
    if demo1_example is not None or demo2_example is not None:
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


class PairValidationDataset(Dataset):
    """Dataset for all (pair, validation, ordering) combinations."""

    def __init__(self, all_pairs, candidate_examples, pair_labels, validation_set):
        self.all_pairs = all_pairs
        self.candidate_examples = candidate_examples
        self.pair_labels = pair_labels
        self.validation_set = validation_set

        # Create all (pair_idx, val_idx, ordering) combinations
        self.items = []
        for pair_idx in range(len(all_pairs)):
            for val_idx in range(len(validation_set)):
                # Two orderings per (pair, validation) combination
                self.items.append((pair_idx, val_idx, 0))  # ordering 0: A=yw, B=yl
                self.items.append((pair_idx, val_idx, 1))  # ordering 1: A=yl, B=yw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pair_idx, val_idx, ordering = self.items[idx]

        idx1, idx2 = self.all_pairs[pair_idx]
        demo1_example = self.candidate_examples[idx1]
        demo2_example = self.candidate_examples[idx2]
        demo1_flip, demo2_flip = self.pair_labels[pair_idx]

        val_example = self.validation_set[val_idx]

        return {
            'pair_idx': pair_idx,
            'val_idx': val_idx,
            'ordering': ordering,
            'demo1_prompt': demo1_example['prompt'],
            'demo1_yw': demo1_example['all_generated_responses'][0],
            'demo1_yl': demo1_example['all_generated_responses'][1],
            'demo1_flip': demo1_flip,
            'demo2_prompt': demo2_example['prompt'],
            'demo2_yw': demo2_example['all_generated_responses'][0],
            'demo2_yl': demo2_example['all_generated_responses'][1],
            'demo2_flip': demo2_flip,
            'val_prompt': val_example['prompt'],
            'val_yw': val_example['all_generated_responses'][0],
            'val_yl': val_example['all_generated_responses'][1],
        }


def collate_fn(batch):
    """Collate function."""
    return batch


def compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch):
    """
    Compute log probabilities for a batch of (pair, validation, ordering) examples.

    Returns: List of dicts with pair_idx, val_idx, ordering, and log_prob
    """
    all_prompts = []
    metadata = []

    for item in batch:
        demo1_example = {
            'prompt': item['demo1_prompt'],
            'all_generated_responses': [item['demo1_yw'], item['demo1_yl']]
        }
        demo2_example = {
            'prompt': item['demo2_prompt'],
            'all_generated_responses': [item['demo2_yw'], item['demo2_yl']]
        }

        val_prompt = item['val_prompt']
        val_yw = item['val_yw']
        val_yl = item['val_yl']
        ordering = item['ordering']

        # Create prompt based on ordering
        if ordering == 0:
            # A=yw, B=yl -> correct is A
            prompt = create_prompt_template(
                val_prompt, val_yw, val_yl,
                demo1_example=demo1_example, demo1_flip=item['demo1_flip'],
                demo2_example=demo2_example, demo2_flip=item['demo2_flip'])
            correct_token = 0  # A
        else:
            # A=yl, B=yw -> correct is B
            prompt = create_prompt_template(
                val_prompt, val_yl, val_yw,
                demo1_example=demo1_example, demo1_flip=item['demo1_flip'],
                demo2_example=demo2_example, demo2_flip=item['demo2_flip'])
            correct_token = 1  # B

        all_prompts.append(prompt)
        metadata.append({
            'pair_idx': item['pair_idx'],
            'val_idx': item['val_idx'],
            'ordering': ordering,
            'correct_token': correct_token
        })

    # Tokenize all prompts at once
    inputs = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
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
            'pair_idx': meta['pair_idx'],
            'val_idx': meta['val_idx'],
            'ordering': meta['ordering'],
            'log_prob': log_prob_correct
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test context pairs with SNR selection")
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
                       help="Path to .npy file with ICL gain results (num_training, num_validation, 6)")
    parser.add_argument("--output_dir", type=str,
                       default="./icl_snr_results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for label assignment")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top context pairs to select and test")
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

    # Initialize accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
        logger.info(f"Testing context pairs with SNR selection")
        logger.info(f"Loading model: {args.model_name}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Batch size per device: {args.batch_size}")
        logger.info(f"Random seed: {args.seed}")
        logger.info(f"Top K: {args.top_k}")

    # Load ICL gain results on all processes for consistency
    icl_gain_results = np.load(args.icl_gain_results_file)

    if accelerator.is_main_process:
        logger.info(f"\nICL gain results shape: {icl_gain_results.shape}")

        # Compute best and worst demonstration indices (5 each = 10 total)
        logger.info(f"Computing top 5 best and worst demonstrations using {args.rank_by}...")
        best_demo_indices, worst_demo_indices, metrics = compute_best_worst_demo_indices(
            icl_gain_results, top_k=5, rank_by=args.rank_by)

        # Combine to get 10 candidate demonstrations
        candidate_demo_indices = best_demo_indices + worst_demo_indices
        num_candidates = len(candidate_demo_indices)

        logger.info(f"\nBest 5 demonstrations: {best_demo_indices}")
        logger.info(f"Worst 5 demonstrations: {worst_demo_indices}")
        logger.info(f"Number of candidate demonstrations: {num_candidates}")
        logger.info(f"Number of pairs: {num_candidates * (num_candidates - 1)}")

    # Broadcast to all processes
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            broadcast_data = {
                'candidate_demo_indices': candidate_demo_indices,
                'num_candidates': num_candidates,
                'best_demo_indices': best_demo_indices,
                'worst_demo_indices': worst_demo_indices
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

    # Select candidate examples using the computed indices (5 best + 5 worst)
    candidate_examples = [training_examples[idx] for idx in candidate_demo_indices]

    # Generate all ordered pairs (i, j) where i != j
    all_pairs = [(i, j) for i in range(num_candidates)
                 for j in range(num_candidates) if i != j]
    num_pairs = len(all_pairs)

    # For each pair, randomly assign one demo to be A and the other to be B
    pair_labels = []
    for pair_idx in range(num_pairs):
        if random.random() < 0.5:
            demo1_flip = False  # demo1 gets label A
            demo2_flip = True   # demo2 gets label B
        else:
            demo1_flip = True   # demo1 gets label B
            demo2_flip = False  # demo2 gets label A
        pair_labels.append((demo1_flip, demo2_flip))

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 1: Testing {num_pairs} pairs on validation set")
        logger.info(f"{'='*80}")

    # Create dataset
    pair_val_dataset = PairValidationDataset(all_pairs, candidate_examples, pair_labels, validation_set)

    if accelerator.is_main_process:
        logger.info(f"Total examples in dataset: {len(pair_val_dataset)} ({num_pairs} pairs × {len(validation_set)} val × 2 orderings)")

    # Create dataloader
    dataloader = DataLoader(
        pair_val_dataset,
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

    # Gather results from all processes
    if accelerator.num_processes > 1:
        gathered_results = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered_results, all_results)
        if accelerator.is_main_process:
            all_results = []
            for results_list in gathered_results:
                all_results.extend(results_list)

    # Only main process continues
    if accelerator.is_main_process:
        logger.info(f"Gathered {len(all_results)} results")

        # Reconstruct result array: (num_pairs, num_val, 2 orderings)
        log_probs_raw = np.zeros((num_pairs, len(validation_set), 2))

        for result in all_results:
            pair_idx = result['pair_idx']
            val_idx = result['val_idx']
            ordering = result['ordering']
            log_probs_raw[pair_idx, val_idx, ordering] = result['log_prob']

        # Average probabilities across orderings, then convert back to log prob
        probs_raw = np.exp(log_probs_raw)
        avg_probs = probs_raw.mean(axis=2)  # (num_pairs, num_val)
        log_probs_with_context = np.log(avg_probs + 1e-10)

        logger.info(f"Log probs with context shape: {log_probs_with_context.shape}")

        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 2: Computing no-context baseline on validation set")
        logger.info(f"{'='*80}")

        # Compute no-context baseline (use same Dataset but with None for demos)
        # For simplicity, compute on main process only
        baseline_results = []
        for val_idx in range(len(validation_set)):
            val_example = validation_set[val_idx]
            val_yw = val_example['all_generated_responses'][0]
            val_yl = val_example['all_generated_responses'][1]
            val_prompt = val_example['prompt']

            # Two orderings
            prompts = [
                create_prompt_template(val_prompt, val_yw, val_yl, None, False, None, False),
                create_prompt_template(val_prompt, val_yl, val_yw, None, False, None, False)
            ]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
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
        gain = log_probs_with_context - log_probs_no_context[np.newaxis, :]  # (num_pairs, 50)

        # For each pair, compute mean and std of gain
        mean_gain = gain.mean(axis=1)  # (num_pairs,)
        std_gain = gain.std(axis=1)    # (num_pairs,)

        # Compute SNR
        snr = mean_gain / (std_gain + 1e-10)  # (num_pairs,)

        # Select top K pairs by SNR
        top_k_indices = np.argsort(snr)[-args.top_k:][::-1]

        logger.info(f"\nTop {args.top_k} pairs by SNR:")
        for rank, pair_idx in enumerate(top_k_indices):
            idx1, idx2 = all_pairs[pair_idx]
            demo1_flip, demo2_flip = pair_labels[pair_idx]
            demo1_label = 'B' if demo1_flip else 'A'
            demo2_label = 'B' if demo2_flip else 'A'
            demo1_train_idx = candidate_demo_indices[idx1]
            demo2_train_idx = candidate_demo_indices[idx2]
            logger.info(f"  Rank {rank+1}: Pair {pair_idx}")
            logger.info(f"    Demo1: candidate {idx1} (train_idx={demo1_train_idx}, label={demo1_label})")
            logger.info(f"    Demo2: candidate {idx2} (train_idx={demo2_train_idx}, label={demo2_label})")
            logger.info(f"    SNR: {snr[pair_idx]:.4f}, Mean gain: {mean_gain[pair_idx]:.4f}, Std gain: {std_gain[pair_idx]:.4f}")

        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 4: Testing top {args.top_k} pairs on test set")
        logger.info(f"{'='*80}")

        # Create test dataset for top-k pairs only
        top_k_pairs = [all_pairs[idx] for idx in top_k_indices]
        top_k_labels = [pair_labels[idx] for idx in top_k_indices]

        test_dataset = PairValidationDataset(top_k_pairs, candidate_examples, top_k_labels, test_set)

        logger.info(f"Test set examples: {len(test_dataset)} ({args.top_k} pairs × {len(test_set)} test × 2 orderings)")

    # Create test dataloader (need to recreate on all processes)
    if accelerator.is_main_process:
        # Broadcast test info
        test_info = {
            'top_k_pairs': top_k_pairs,
            'top_k_labels': top_k_labels
        }
    else:
        test_info = None

    if accelerator.num_processes > 1:
        test_info_list = [test_info]
        torch.distributed.broadcast_object_list(test_info_list, src=0)
        if not accelerator.is_main_process:
            top_k_pairs = test_info_list[0]['top_k_pairs']
            top_k_labels = test_info_list[0]['top_k_labels']

    test_dataset = PairValidationDataset(top_k_pairs, candidate_examples, top_k_labels, test_set)

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
        log_probs_test_raw = np.zeros((args.top_k, len(test_set), 2))

        for result in test_results_all:
            pair_idx = result['pair_idx']  # This is now in range [0, top_k-1]
            val_idx = result['val_idx']
            ordering = result['ordering']
            log_probs_test_raw[pair_idx, val_idx, ordering] = result['log_prob']

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
                create_prompt_template(val_prompt, val_yw, val_yl, None, False, None, False),
                create_prompt_template(val_prompt, val_yl, val_yw, None, False, None, False)
            ]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
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
        for rank, pair_idx in enumerate(top_k_indices):
            idx1, idx2 = all_pairs[pair_idx]
            demo1_flip, demo2_flip = pair_labels[pair_idx]
            demo1_label = 'B' if demo1_flip else 'A'
            demo2_label = 'B' if demo2_flip else 'A'
            demo1_train_idx = candidate_demo_indices[idx1]
            demo2_train_idx = candidate_demo_indices[idx2]
            mean_test_gain = test_gain[rank, :].mean()
            std_test_gain = test_gain[rank, :].std()
            test_snr = mean_test_gain / (std_test_gain + 1e-10)

            logger.info(f"  Rank {rank+1}: Pair {pair_idx}")
            logger.info(f"    Demo1: candidate {idx1} (train_idx={demo1_train_idx}, label={demo1_label})")
            logger.info(f"    Demo2: candidate {idx2} (train_idx={demo2_train_idx}, label={demo2_label})")
            logger.info(f"    Test SNR: {test_snr:.4f}, Mean gain: {mean_test_gain:.4f}, Std gain: {std_test_gain:.4f}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"icl_snr_pairs_results_{actual_persona_id}.npz")

        np.savez(output_file,
                 log_probs_no_context=log_probs_no_context,
                 log_probs_with_context=log_probs_with_context,
                 gain=gain,
                 mean_gain=mean_gain,
                 std_gain=std_gain,
                 snr=snr,
                 top_k_indices=top_k_indices,
                 all_pairs=all_pairs,
                 pair_labels=pair_labels,
                 log_probs_test_no_context=log_probs_test_no_context,
                 log_probs_test_with_context=log_probs_test_with_context,
                 test_gain=test_gain,
                 candidate_demo_indices=candidate_demo_indices,
                 best_demo_indices=best_demo_indices,
                 worst_demo_indices=worst_demo_indices,
                 rank_by=args.rank_by)

        logger.info(f"\nResults saved to: {output_file}")
        logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
