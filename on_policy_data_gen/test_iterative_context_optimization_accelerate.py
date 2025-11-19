#!/usr/bin/env python3
"""
Iterative context optimization algorithm.

1. Load top_n best contexts from validation results
2. For each context, iteratively optimize by replacing demonstrations:
   - For each of k positions, try n_candidates test examples × 2 orientations
   - Evaluate each candidate on validation set (50 examples × 2 orderings)
   - Keep replacements that improve validation log probability
   - Maintain label balance (e.g., 2 A's and 2 B's for 4-shot)
3. Test optimized contexts on full test set

Algorithm details:
- 4 positions to optimize × 2 orientations × 100 candidates = 800 evaluations per position
- Each evaluation: 50 validation examples × 2 orderings = 100 forward passes
- Total: 4 × 800 × 100 = 320,000 forward passes per context
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

# Set NCCL timeout (in seconds) - increase for long-running evaluations
# Default is 600s (10 min), increase to 2 hours for iterative optimization
os.environ.setdefault('NCCL_TIMEOUT', '7200')
os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # Enable blocking wait for better error messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt_template(question, response_a, response_b, demo_examples=None, demo_flips=None):
    """
    Create a prompt for pairwise comparison with k demonstration contexts.
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


class ValidationDataset(Dataset):
    """Dataset for evaluating a single context on validation set."""

    def __init__(self, demo_indices, demo_labels, candidate_examples, validation_set):
        """
        Args:
            demo_indices: List of k demonstration indices
            demo_labels: List of k boolean flip flags
            candidate_examples: All available examples (training + test subset)
            validation_set: Validation examples
        """
        self.demo_indices = demo_indices
        self.demo_labels = demo_labels
        self.candidate_examples = candidate_examples
        self.validation_set = validation_set

        # Create all (val_idx, ordering) combinations
        self.items = []
        for val_idx in range(len(validation_set)):
            self.items.append((val_idx, 0))  # ordering 0: A=yw, B=yl
            self.items.append((val_idx, 1))  # ordering 1: A=yl, B=yw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        val_idx, ordering = self.items[idx]
        val_example = self.validation_set[val_idx]

        # Prepare demonstration data
        demo_data = []
        for demo_idx, demo_flip in zip(self.demo_indices, self.demo_labels):
            demo_example = self.candidate_examples[demo_idx]
            demo_data.append({
                'prompt': demo_example['prompt'],
                'yw': demo_example['all_generated_responses'][0],
                'yl': demo_example['all_generated_responses'][1],
                'flip': demo_flip
            })

        return {
            'val_idx': val_idx,
            'ordering': ordering,
            'demo_data': demo_data,
            'val_prompt': val_example['prompt'],
            'val_yw': val_example['all_generated_responses'][0],
            'val_yl': val_example['all_generated_responses'][1],
        }


class TestDataset(Dataset):
    """Dataset for testing optimized contexts on test set."""

    def __init__(self, optimized_contexts, candidate_examples, test_set):
        """
        Args:
            optimized_contexts: List of (demo_indices, demo_labels) tuples
            candidate_examples: All available examples
            test_set: Test examples
        """
        self.optimized_contexts = optimized_contexts
        self.candidate_examples = candidate_examples
        self.test_set = test_set

        # Create all (context_idx, test_idx, ordering) combinations
        self.items = []
        for ctx_idx in range(len(optimized_contexts)):
            for test_idx in range(len(test_set)):
                self.items.append((ctx_idx, test_idx, 0))
                self.items.append((ctx_idx, test_idx, 1))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ctx_idx, test_idx, ordering = self.items[idx]
        demo_indices, demo_labels = self.optimized_contexts[ctx_idx]
        test_example = self.test_set[test_idx]

        # Prepare demonstration data
        demo_data = []
        for demo_idx, demo_flip in zip(demo_indices, demo_labels):
            demo_example = self.candidate_examples[demo_idx]
            demo_data.append({
                'prompt': demo_example['prompt'],
                'yw': demo_example['all_generated_responses'][0],
                'yl': demo_example['all_generated_responses'][1],
                'flip': demo_flip
            })

        return {
            'ctx_idx': ctx_idx,
            'test_idx': test_idx,
            'ordering': ordering,
            'demo_data': demo_data,
            'test_prompt': test_example['prompt'],
            'test_yw': test_example['all_generated_responses'][0],
            'test_yl': test_example['all_generated_responses'][1],
        }


def collate_fn(batch):
    """Collate function."""
    return batch


def compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch):
    """Compute log probabilities for a batch."""
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

        if 'val_prompt' in item:
            # Validation item
            prompt_key = 'val_prompt'
            yw_key = 'val_yw'
            yl_key = 'val_yl'
        else:
            # Test item
            prompt_key = 'test_prompt'
            yw_key = 'test_yw'
            yl_key = 'test_yl'

        test_prompt = item[prompt_key]
        test_yw = item[yw_key]
        test_yl = item[yl_key]
        ordering = item['ordering']

        # Create prompt based on ordering
        if ordering == 0:
            prompt = create_prompt_template(
                test_prompt, test_yw, test_yl,
                demo_examples=demo_examples,
                demo_flips=demo_flips)
            correct_token = 0  # A
        else:
            prompt = create_prompt_template(
                test_prompt, test_yl, test_yw,
                demo_examples=demo_examples,
                demo_flips=demo_flips)
            correct_token = 1  # B

        all_prompts.append(prompt)
        meta = {
            'ordering': ordering,
            'correct_token': correct_token
        }
        if 'val_idx' in item:
            meta['val_idx'] = item['val_idx']
        if 'ctx_idx' in item:
            meta['ctx_idx'] = item['ctx_idx']
        if 'test_idx' in item:
            meta['test_idx'] = item['test_idx']
        metadata.append(meta)

    # Tokenize all prompts at once
    inputs = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        logit_a = logits[:, token_a_id]
        logit_b = logits[:, token_b_id]
        logits_ab = torch.stack([logit_a, logit_b], dim=1)
        log_probs_ab = F.log_softmax(logits_ab, dim=1)

    # Extract results
    results = []
    for i, meta in enumerate(metadata):
        log_prob_correct = log_probs_ab[i, meta['correct_token']].item()
        result = {'log_prob': log_prob_correct, 'ordering': meta['ordering']}
        if 'val_idx' in meta:
            result['val_idx'] = meta['val_idx']
        if 'ctx_idx' in meta:
            result['ctx_idx'] = meta['ctx_idx']
        if 'test_idx' in meta:
            result['test_idx'] = meta['test_idx']
        results.append(result)

    return results


def evaluate_context_on_validation(model, tokenizer, token_a_id, token_b_id,
                                   demo_indices, demo_labels, candidate_examples,
                                   validation_set, batch_size, accelerator):
    """Evaluate a single context on validation set and return mean log prob."""
    dataset = ValidationDataset(demo_indices, demo_labels, candidate_examples, validation_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    dataloader = accelerator.prepare(dataloader)

    all_results = []
    for batch in dataloader:
        batch_results = compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch)
        all_results.extend(batch_results)

    # Gather results
    if accelerator.num_processes > 1:
        gathered_results = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered_results, all_results)
        if accelerator.is_main_process:
            all_results = []
            for results_list in gathered_results:
                all_results.extend(results_list)

    if accelerator.is_main_process:
        # Compute mean log prob
        log_probs_raw = np.zeros((len(validation_set), 2))
        for result in all_results:
            val_idx = result['val_idx']
            ordering = result['ordering']
            log_probs_raw[val_idx, ordering] = result['log_prob']

        probs_raw = np.exp(log_probs_raw)
        avg_probs = probs_raw.mean(axis=1)
        log_probs = np.log(avg_probs + 1e-10)
        mean_log_prob = log_probs.mean()
        return mean_log_prob
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative context optimization")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to use for evaluation")
    parser.add_argument("--dataset_name", type=str,
                       default="sher222/persona-iterative-responses",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--persona_id", type=str,
                       default="cdf7cefb-7341-41d7-a193-ff0f2f962cf9",
                       help="Specific persona UUID to filter for")
    parser.add_argument("--validation_results_file", type=str,
                       required=True,
                       help="Path to .npz file with validation results")
    parser.add_argument("--output_dir", type=str,
                       default="./icl_optimized_results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--top_n", type=int, default=5,
                       help="Number of best contexts to optimize")
    parser.add_argument("--n_candidates", type=int, default=100,
                       help="Number of test examples to try as replacements")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize accelerator with extended timeout for long-running optimization
    from datetime import timedelta
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # 2 hours timeout
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    if accelerator.is_main_process:
        logger.info(f"Iterative context optimization")
        logger.info(f"Loading model: {args.model_name}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Batch size per device: {args.batch_size}")

    # Load validation results
    if accelerator.is_main_process:
        logger.info(f"\nLoading validation results from: {args.validation_results_file}")

    validation_results = np.load(args.validation_results_file)

    if accelerator.is_main_process:
        # Extract data
        candidate_demo_indices = validation_results['candidate_demo_indices'].tolist()
        all_tuples = [tuple(t) for t in validation_results['all_tuples']]
        tuple_labels = [tuple(t) for t in validation_results['tuple_labels']]
        snr = validation_results['snr']
        k = int(validation_results['k'])

        logger.info(f"Loaded {len(all_tuples)} tuples, k={k}")
        logger.info(f"Candidate demonstrations from training: {candidate_demo_indices}")

        # Select top_n best contexts
        best_indices = np.argsort(snr)[-args.top_n:][::-1].tolist()
        initial_contexts = [(list(all_tuples[i]), list(tuple_labels[i])) for i in best_indices]
        initial_snrs = [snr[i] for i in best_indices]

        logger.info(f"\nSelected top {args.top_n} contexts for optimization:")
        for i, (ctx, snr_val) in enumerate(zip(initial_contexts, initial_snrs)):
            logger.info(f"  Context {i}: indices={ctx[0]}, labels={ctx[1]}, SNR: {snr_val:.4f}")

    # Broadcast to all processes
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            broadcast_data = {
                'candidate_demo_indices': candidate_demo_indices,
                'initial_contexts': initial_contexts,
                'k': k
            }
        else:
            broadcast_data = None

        import torch.distributed as dist
        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        if not accelerator.is_main_process:
            candidate_demo_indices = broadcast_list[0]['candidate_demo_indices']
            initial_contexts = broadcast_list[0]['initial_contexts']
            k = broadcast_list[0]['k']

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model = accelerator.prepare(model)

    token_a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    token_b_id = tokenizer.encode("B", add_special_tokens=False)[0]

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train")
    filtered_dataset, actual_persona_id = filter_dataset_by_persona(dataset, args.persona_id)
    persona_data = prepare_persona_dataset_as_pairwise(filtered_dataset)

    test_set = persona_data[:512]
    validation_set = persona_data[512:562]
    training_examples = persona_data[562:]

    if accelerator.is_main_process:
        logger.info(f"\nTest set: {len(test_set)} examples")
        logger.info(f"Validation set: {len(validation_set)} examples")
        logger.info(f"Training examples: {len(training_examples)} examples")

        # Select random subset of test examples as replacement candidates
        test_subset_indices = random.sample(range(len(test_set)), args.n_candidates)
        test_subset = [test_set[i] for i in test_subset_indices]
        logger.info(f"Selected {len(test_subset)} test examples as replacement candidates")

        # Create candidate pool: training examples + test subset
        # Assign indices: 0 to len(training)-1 are training, len(training) onwards are test
        candidate_examples = training_examples + test_subset
        training_offset = 0
        test_offset = len(training_examples)

        logger.info(f"Total candidate pool size: {len(candidate_examples)}")
        logger.info(f"  Training examples: indices 0-{len(training_examples)-1}")
        logger.info(f"  Test subset: indices {test_offset}-{len(candidate_examples)-1}")

    # Broadcast
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            broadcast_data = {
                'candidate_examples': candidate_examples,
                'test_offset': test_offset,
                'test_subset_indices': test_subset_indices
            }
        else:
            broadcast_data = None

        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        if not accelerator.is_main_process:
            candidate_examples = broadcast_list[0]['candidate_examples']
            test_offset = broadcast_list[0]['test_offset']
            test_subset_indices = broadcast_list[0]['test_subset_indices']

    # Optimization loop
    optimized_contexts = []

    for ctx_num, (demo_indices, demo_labels) in enumerate(initial_contexts):
        if accelerator.is_main_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"Optimizing context {ctx_num + 1}/{args.top_n}")
            logger.info(f"{'='*80}")
            logger.info(f"Initial: indices={demo_indices}, labels={demo_labels}")

        # Evaluate initial context (ALL processes participate)
        initial_score = evaluate_context_on_validation(
            model, tokenizer, token_a_id, token_b_id,
            demo_indices, demo_labels, candidate_examples,
            validation_set, args.batch_size, accelerator)

        if accelerator.is_main_process:
            logger.info(f"Initial validation score: {initial_score:.4f}")

        current_indices = demo_indices.copy()
        current_labels = demo_labels.copy()
        current_score = initial_score

        # Optimize each position
        for pos in range(k):
            if accelerator.is_main_process:
                logger.info(f"\n  Optimizing position {pos}...")
                logger.info(f"    Current label at position {pos}: {'B' if current_labels[pos] else 'A'}")

            best_replacement_idx = current_indices[pos]
            best_replacement_label = current_labels[pos]
            best_replacement_score = current_score

            # Create list of candidates to try
            if accelerator.is_main_process:
                # Try test examples as replacements
                # We maintain the label balance by only trying the same label orientation
                flip = current_labels[pos]  # Keep the same label to maintain balance

                candidates_to_try = []
                for test_idx in range(test_offset, len(candidate_examples)):
                    candidate_indices = current_indices.copy()
                    candidate_labels = current_labels.copy()
                    candidate_indices[pos] = test_idx
                    candidates_to_try.append((candidate_indices, candidate_labels, test_idx))

            # Broadcast candidates to all processes
            if accelerator.num_processes > 1:
                if accelerator.is_main_process:
                    broadcast_data = {'candidates_to_try': candidates_to_try, 'pos': pos, 'flip': current_labels[pos]}
                else:
                    broadcast_data = None

                import torch.distributed as dist
                broadcast_list = [broadcast_data]
                dist.broadcast_object_list(broadcast_list, src=0)
                if not accelerator.is_main_process:
                    candidates_to_try = broadcast_list[0]['candidates_to_try']
                    pos = broadcast_list[0]['pos']
                    flip = broadcast_list[0]['flip']

            # All processes evaluate candidates
            for candidate_indices, candidate_labels, test_idx in tqdm(candidates_to_try,
                                    desc=f"    Trying replacements at position {pos}" if accelerator.is_main_process else None,
                                    disable=not accelerator.is_main_process):
                # ALL processes evaluate
                candidate_score = evaluate_context_on_validation(
                    model, tokenizer, token_a_id, token_b_id,
                    candidate_indices, candidate_labels, candidate_examples,
                    validation_set, args.batch_size, accelerator)

                if accelerator.is_main_process:
                    if candidate_score > best_replacement_score:
                        best_replacement_idx = test_idx
                        best_replacement_label = flip  # Same as current
                        best_replacement_score = candidate_score

            if accelerator.is_main_process:
                # Update if improvement found
                if best_replacement_score > current_score:
                    logger.info(f"    Found improvement! {current_score:.4f} -> {best_replacement_score:.4f}")
                    logger.info(f"    Replaced index {current_indices[pos]} with {best_replacement_idx}")
                    current_indices[pos] = best_replacement_idx
                    current_labels[pos] = best_replacement_label
                    current_score = best_replacement_score
                else:
                    logger.info(f"    No improvement found, keeping position {pos} unchanged")

        if accelerator.is_main_process:
            logger.info(f"\nOptimization complete for context {ctx_num + 1}")
            logger.info(f"  Initial: indices={demo_indices}, labels={demo_labels}")
            logger.info(f"  Final:   indices={current_indices}, labels={current_labels}")
            logger.info(f"  Score improvement: {initial_score:.4f} -> {current_score:.4f}")

            optimized_contexts.append((current_indices, current_labels))

        accelerator.wait_for_everyone()

    # Test on full test set
    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing optimized contexts on full test set")
        logger.info(f"{'='*80}")

    # Broadcast optimized contexts
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            broadcast_data = {'optimized_contexts': optimized_contexts}
        else:
            broadcast_data = None

        broadcast_list = [broadcast_data]
        dist.broadcast_object_list(broadcast_list, src=0)
        if not accelerator.is_main_process:
            optimized_contexts = broadcast_list[0]['optimized_contexts']

    test_dataset = TestDataset(optimized_contexts, candidate_examples, test_set)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_dataloader = accelerator.prepare(test_dataloader)

    test_results = []
    for batch in tqdm(test_dataloader, desc="Testing", disable=not accelerator.is_main_process):
        batch_results = compute_log_prob_batch(model, tokenizer, token_a_id, token_b_id, batch)
        test_results.extend(batch_results)

    # Gather results
    if accelerator.num_processes > 1:
        gathered_results = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered_results, test_results)
        if accelerator.is_main_process:
            test_results = []
            for results_list in gathered_results:
                test_results.extend(results_list)

    if accelerator.is_main_process:
        # Process test results
        log_probs_raw = np.zeros((args.top_n, len(test_set), 2))
        for result in test_results:
            ctx_idx = result['ctx_idx']
            test_idx = result['test_idx']
            ordering = result['ordering']
            log_probs_raw[ctx_idx, test_idx, ordering] = result['log_prob']

        probs_raw = np.exp(log_probs_raw)
        avg_probs = probs_raw.mean(axis=2)
        log_probs_with_context = np.log(avg_probs + 1e-10)

        # Compute no-context baseline
        logger.info("Computing no-context baseline...")
        baseline_results = []
        for test_example in test_set:
            prompts = [
                create_prompt_template(test_example['prompt'],
                                     test_example['all_generated_responses'][0],
                                     test_example['all_generated_responses'][1],
                                     None, None),
                create_prompt_template(test_example['prompt'],
                                     test_example['all_generated_responses'][1],
                                     test_example['all_generated_responses'][0],
                                     None, None)
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

            prob_1 = np.exp(log_probs_ab[0, 0].item())
            prob_2 = np.exp(log_probs_ab[1, 1].item())
            baseline_results.append(np.log((prob_1 + prob_2) / 2.0 + 1e-10))

        log_probs_no_context = np.array(baseline_results)
        gain = log_probs_with_context - log_probs_no_context[np.newaxis, :]

        # Compute accuracy
        accuracy_with_context = (avg_probs > 0.5).astype(float)
        probs_no_context = np.exp(log_probs_no_context)
        accuracy_no_context = (probs_no_context > 0.5).astype(float)
        accuracy_gain = accuracy_with_context - accuracy_no_context[np.newaxis, :]

        logger.info(f"\n{'='*80}")
        logger.info(f"Test set results:")
        logger.info(f"{'='*80}")
        for i, (indices, labels) in enumerate(optimized_contexts):
            logger.info(f"\nContext {i+1}:")
            logger.info(f"  Indices: {indices}")
            logger.info(f"  Labels: {labels}")
            logger.info(f"  Mean log prob: {log_probs_with_context[i].mean():.4f}")
            logger.info(f"  Mean gain: {gain[i].mean():.4f}")
            logger.info(f"  Mean accuracy: {accuracy_with_context[i].mean():.4f}")
            logger.info(f"  Mean acc gain: {accuracy_gain[i].mean():.4f}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"optimized_contexts_{actual_persona_id}.npz")

        np.savez(output_file,
                 optimized_contexts=optimized_contexts,
                 initial_contexts=initial_contexts,
                 test_subset_indices=test_subset_indices,
                 log_probs_with_context=log_probs_with_context,
                 log_probs_no_context=log_probs_no_context,
                 gain=gain,
                 accuracy_with_context=accuracy_with_context,
                 accuracy_no_context=accuracy_no_context,
                 accuracy_gain=accuracy_gain,
                 k=k)

        logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
