#!/usr/bin/env python3
"""
Test selected demonstration pairs on test set.

1. Load existing pair validation results from npz file
2. Select 5 best, 5 worst, and 5 random pairs based on validation SNR
3. Test each of the 15 pairs on test set
4. Record log probability, gain, and accuracy on test set
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
    prepare_persona_dataset_as_pairwise
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


class PairTestDataset(Dataset):
    """Dataset for testing demonstration pairs on test set."""

    def __init__(self, selected_pairs, candidate_examples, pair_labels, test_set):
        """
        Args:
            selected_pairs: List of (idx1, idx2) tuples for demonstration pairs
            candidate_examples: List of all candidate demonstration examples
            pair_labels: List of (demo1_flip, demo2_flip) tuples for each pair
            test_set: List of test examples
        """
        self.selected_pairs = selected_pairs
        self.candidate_examples = candidate_examples
        self.pair_labels = pair_labels
        self.test_set = test_set

        # Create all (pair_position, test_idx, ordering) combinations
        self.items = []
        for pair_position in range(len(selected_pairs)):
            for test_idx in range(len(test_set)):
                # Two orderings per (pair, test) combination
                self.items.append((pair_position, test_idx, 0))  # ordering 0: A=yw, B=yl
                self.items.append((pair_position, test_idx, 1))  # ordering 1: A=yl, B=yw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pair_position, test_idx, ordering = self.items[idx]

        idx1, idx2 = self.selected_pairs[pair_position]
        demo1_example = self.candidate_examples[idx1]
        demo2_example = self.candidate_examples[idx2]
        demo1_flip, demo2_flip = self.pair_labels[pair_position]

        test_example = self.test_set[test_idx]

        return {
            'pair_position': pair_position,  # Position in selected pairs list (0-14)
            'test_idx': test_idx,
            'ordering': ordering,
            'demo1_prompt': demo1_example['prompt'],
            'demo1_yw': demo1_example['all_generated_responses'][0],
            'demo1_yl': demo1_example['all_generated_responses'][1],
            'demo1_flip': demo1_flip,
            'demo2_prompt': demo2_example['prompt'],
            'demo2_yw': demo2_example['all_generated_responses'][0],
            'demo2_yl': demo2_example['all_generated_responses'][1],
            'demo2_flip': demo2_flip,
            'test_prompt': test_example['prompt'],
            'test_yw': test_example['all_generated_responses'][0],
            'test_yl': test_example['all_generated_responses'][1],
        }


def collate_fn(batch):
    """Collate function."""
    return batch


def compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch):
    """
    Compute log probabilities for a batch of (pair, test, ordering) examples.

    Returns: List of dicts with pair_position, test_idx, ordering, and log_prob
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

        test_prompt = item['test_prompt']
        test_yw = item['test_yw']
        test_yl = item['test_yl']
        ordering = item['ordering']

        # Create prompt based on ordering
        if ordering == 0:
            # A=yw, B=yl -> correct is A
            prompt = create_prompt_template(
                test_prompt, test_yw, test_yl,
                demo1_example=demo1_example, demo1_flip=item['demo1_flip'],
                demo2_example=demo2_example, demo2_flip=item['demo2_flip'])
            correct_token = 0  # A
        else:
            # A=yl, B=yw -> correct is B
            prompt = create_prompt_template(
                test_prompt, test_yl, test_yw,
                demo1_example=demo1_example, demo1_flip=item['demo1_flip'],
                demo2_example=demo2_example, demo2_flip=item['demo2_flip'])
            correct_token = 1  # B

        all_prompts.append(prompt)
        metadata.append({
            'pair_position': item['pair_position'],
            'test_idx': item['test_idx'],
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
            'pair_position': meta['pair_position'],
            'test_idx': meta['test_idx'],
            'ordering': meta['ordering'],
            'log_prob': log_prob_correct
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test selected demonstration pairs on test set")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to use for evaluation")
    parser.add_argument("--dataset_name", type=str,
                       default="sher222/persona-iterative-responses",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--persona_id", type=str,
                       default="cdf7cefb-7341-41d7-a193-ff0f2f962cf9",
                       help="Specific persona UUID to filter for")
    parser.add_argument("--pair_results_file", type=str,
                       required=True,
                       help="Path to .npz file with pair validation results")
    parser.add_argument("--output_dir", type=str,
                       default="./icl_pair_test_results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for random pair selection")
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
        logger.info(f"Testing selected demonstration pairs on test set")
        logger.info(f"Loading model: {args.model_name}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Batch size per device: {args.batch_size}")
        logger.info(f"Random seed: {args.seed}")

    # Load pair validation results from npz file
    if accelerator.is_main_process:
        logger.info(f"\nLoading pair validation results from: {args.pair_results_file}")

    pair_results = np.load(args.pair_results_file)

    if accelerator.is_main_process:
        # Extract data from npz file
        candidate_demo_indices = pair_results['candidate_demo_indices'].tolist()
        all_pairs = [tuple(p) for p in pair_results['all_pairs']]
        pair_labels = [tuple(p) for p in pair_results['pair_labels']]
        snr = pair_results['snr']

        logger.info(f"Loaded {len(all_pairs)} pairs from validation results")
        logger.info(f"Candidate demonstrations: {candidate_demo_indices}")
        logger.info(f"SNR range: [{snr.min():.4f}, {snr.max():.4f}]")

        # Select 5 best pairs (highest SNR)
        best_pair_indices = np.argsort(snr)[-5:][::-1].tolist()

        # Select 5 worst pairs (lowest SNR)
        worst_pair_indices = np.argsort(snr)[:5].tolist()

        # Select 5 random pairs from all 90 pairs
        random_pair_indices = random.sample(range(len(all_pairs)), 5)

        # Combine to get 15 pairs
        selected_pair_indices = best_pair_indices + worst_pair_indices + random_pair_indices
        pair_categories = ['best'] * 5 + ['worst'] * 5 + ['random'] * 5

        # Get the actual pairs and labels
        selected_pairs = [all_pairs[i] for i in selected_pair_indices]
        selected_labels = [pair_labels[i] for i in selected_pair_indices]
        selected_snrs = [snr[i] for i in selected_pair_indices]

        logger.info(f"\n{'='*80}")
        logger.info(f"Selected pairs:")
        logger.info(f"{'='*80}")
        logger.info(f"\n{'Category':<10} {'Pair Idx':<10} {'Demo1':<8} {'Demo2':<8} {'Label1':<8} {'Label2':<8} {'SNR':<12}")
        logger.info(f"{'-'*80}")

        for category, pair_idx, (idx1, idx2), (flip1, flip2), pair_snr in zip(
            pair_categories, selected_pair_indices, selected_pairs, selected_labels, selected_snrs):
            label1 = 'B' if flip1 else 'A'
            label2 = 'B' if flip2 else 'A'
            logger.info(f"{category:<10} {pair_idx:<10} {idx1:<8} {idx2:<8} {label1:<8} {label2:<8} {pair_snr:<12.4f}")

    # Broadcast to all processes
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            broadcast_data = {
                'candidate_demo_indices': candidate_demo_indices,
                'selected_pairs': selected_pairs,
                'selected_labels': selected_labels,
                'selected_snrs': selected_snrs,
                'pair_categories': pair_categories,
                'selected_pair_indices': selected_pair_indices
            }
        else:
            broadcast_data = None

        import torch.distributed as dist
        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        if not accelerator.is_main_process:
            candidate_demo_indices = broadcast_list[0]['candidate_demo_indices']
            selected_pairs = broadcast_list[0]['selected_pairs']
            selected_labels = broadcast_list[0]['selected_labels']
            selected_snrs = broadcast_list[0]['selected_snrs']
            pair_categories = broadcast_list[0]['pair_categories']
            selected_pair_indices = broadcast_list[0]['selected_pair_indices']

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

    # Get candidate examples from training set
    candidate_examples = training_examples

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {len(selected_pairs)} pairs on test set")
        logger.info(f"{'='*80}")

    # Create dataset
    test_dataset = PairTestDataset(
        selected_pairs, candidate_examples, selected_labels, test_set)

    if accelerator.is_main_process:
        logger.info(f"Total examples in dataset: {len(test_dataset)} "
                   f"({len(selected_pairs)} pairs × {len(test_set)} test × 2 orderings)")

    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
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
        logger.info("Starting computation on test set...")

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

        # Reconstruct result array: (num_pairs, num_test, 2 orderings)
        num_pairs = len(selected_pairs)
        log_probs_raw = np.zeros((num_pairs, len(test_set), 2))

        for result in all_results:
            pair_position = result['pair_position']
            test_idx = result['test_idx']
            ordering = result['ordering']
            log_probs_raw[pair_position, test_idx, ordering] = result['log_prob']

        # Average probabilities across orderings, then convert back to log prob
        probs_raw = np.exp(log_probs_raw)
        avg_probs = probs_raw.mean(axis=2)  # (num_pairs, num_test)
        log_probs_with_context = np.log(avg_probs + 1e-10)

        logger.info(f"Log probs with context shape: {log_probs_with_context.shape}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Computing no-context baseline on test set")
        logger.info(f"{'='*80}")

        # Compute no-context baseline
        baseline_results = []
        for test_idx in range(len(test_set)):
            test_example = test_set[test_idx]
            test_yw = test_example['all_generated_responses'][0]
            test_yl = test_example['all_generated_responses'][1]
            test_prompt = test_example['prompt']

            # Two orderings
            prompts = [
                create_prompt_template(test_prompt, test_yw, test_yl, None, False, None, False),
                create_prompt_template(test_prompt, test_yl, test_yw, None, False, None, False)
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

        # Compute gain
        gain = log_probs_with_context - log_probs_no_context[np.newaxis, :]  # (num_pairs, num_test)

        # Compute accuracy (probability > 0.5)
        accuracy_with_context = (avg_probs > 0.5).astype(float)

        # No-context accuracy
        probs_no_context = np.exp(log_probs_no_context)
        accuracy_no_context = (probs_no_context > 0.5).astype(float)

        # Accuracy gain
        accuracy_gain = accuracy_with_context - accuracy_no_context[np.newaxis, :]

        # Compute statistics for each pair
        logger.info(f"\n{'='*80}")
        logger.info(f"Results on test set")
        logger.info(f"{'='*80}")

        logger.info(f"\n{'Category':<10} {'Pair Idx':<10} {'Demo1-Demo2':<15} {'Mean Log Prob':<15} {'Mean Gain':<12} {'Mean Acc':<12} {'Acc Gain':<12}")
        logger.info(f"{'-'*100}")

        for i, (category, pair_idx, (idx1, idx2)) in enumerate(zip(pair_categories, selected_pair_indices, selected_pairs)):
            mean_log_prob = log_probs_with_context[i, :].mean()
            mean_gain = gain[i, :].mean()
            std_gain = gain[i, :].std()
            mean_acc = accuracy_with_context[i, :].mean()
            mean_acc_gain = accuracy_gain[i, :].mean()
            pair_str = f"{idx1}-{idx2}"

            logger.info(f"{category:<10} {pair_idx:<10} {pair_str:<15} {mean_log_prob:<15.4f} "
                       f"{mean_gain:<12.4f} {mean_acc:<12.4f} {mean_acc_gain:<12.4f}")

        # Compute category-level statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"Category-level statistics")
        logger.info(f"{'='*80}")

        for category_name in ['best', 'worst', 'random']:
            category_mask = np.array([cat == category_name for cat in pair_categories])
            category_indices = np.where(category_mask)[0]

            category_log_probs = log_probs_with_context[category_indices, :].mean()
            category_gain = gain[category_indices, :].mean()
            category_acc = accuracy_with_context[category_indices, :].mean()
            category_acc_gain = accuracy_gain[category_indices, :].mean()

            logger.info(f"\n{category_name.upper()}:")
            logger.info(f"  Mean log prob: {category_log_probs:.4f}")
            logger.info(f"  Mean gain: {category_gain:.4f}")
            logger.info(f"  Mean accuracy: {category_acc:.4f}")
            logger.info(f"  Mean accuracy gain: {category_acc_gain:.4f}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"icl_pair_test_results_{actual_persona_id}.npz")

        np.savez(output_file,
                 candidate_demo_indices=candidate_demo_indices,
                 selected_pairs=selected_pairs,
                 selected_labels=selected_labels,
                 selected_snrs=selected_snrs,
                 selected_pair_indices=selected_pair_indices,
                 pair_categories=pair_categories,
                 log_probs_with_context=log_probs_with_context,
                 log_probs_no_context=log_probs_no_context,
                 gain=gain,
                 accuracy_with_context=accuracy_with_context,
                 accuracy_no_context=accuracy_no_context,
                 accuracy_gain=accuracy_gain)

        logger.info(f"\nResults saved to: {output_file}")
        logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
