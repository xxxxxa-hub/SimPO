#!/usr/bin/env python3
"""
Test accuracy on 512 test examples using best and worst demonstrations.

For each test example and demonstration (best/worst), we compute accuracy across 4 runs:
1. Demo: A=yw B=yl [[A]], Test: A=yw B=yl - check if model prefers A
2. Demo: A=yw B=yl [[A]], Test: A=yl B=yw - check if model prefers B
3. Demo: A=yl B=yw [[B]], Test: A=yw B=yl - check if model prefers A
4. Demo: A=yl B=yw [[B]], Test: A=yl B=yw - check if model prefers B

Average accuracy across these 4 runs for each test example.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_dataset_by_persona(dataset, persona_id=None):
    """Filter dataset to only include samples from a specific persona."""
    if 'persona_uuid' in dataset.column_names:
        persona_col = 'persona_uuid'
    elif 'score_persona' in dataset.column_names:
        persona_ids = [sample['score_persona'].get('persona_uuid') if isinstance(sample.get('score_persona'), dict) else None
                      for sample in dataset]
        if persona_id is None:
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
    """Convert persona dataset format to pairwise comparison format."""
    converted_data = []
    for sample in dataset:
        prompt = sample['x']
        yw = sample['yw'].strip()
        yl = sample['yl'].strip()

        converted_data.append({
            'prompt': prompt,
            'all_generated_responses': [yw, yl],
        })

    return converted_data


def create_prompt_template(question, response_a, response_b, context_example=None, context_flip=False):
    """Create a prompt for pairwise comparison with optional context example."""
    prompt_text = ""

    # Add context example if provided
    if context_example is not None:
        ctx_question = context_example['prompt']
        ctx_yw = context_example['all_generated_responses'][0]
        ctx_yl = context_example['all_generated_responses'][1]

        prompt_text += "# Example\n"
        prompt_text += f"## Question\n{ctx_question}\n\n"

        if context_flip:
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
    if context_example is not None:
        prompt_text += "Given the example above, evaluate the quality of two AI assistants' responses.\n\n"
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


class TestDataset(Dataset):
    """Dataset for testing with a specific demonstration."""

    def __init__(self, test_examples, demo_example):
        self.test_examples = test_examples
        self.demo_example = demo_example

    def __len__(self):
        return len(self.test_examples)

    def __getitem__(self, idx):
        test_example = self.test_examples[idx]

        return {
            'test_idx': idx,
            'demo_prompt': self.demo_example['prompt'],
            'demo_yw': self.demo_example['all_generated_responses'][0],
            'demo_yl': self.demo_example['all_generated_responses'][1],
            'test_prompt': test_example['prompt'],
            'test_yw': test_example['all_generated_responses'][0],
            'test_yl': test_example['all_generated_responses'][1],
        }


def collate_fn(batch):
    """Collate function."""
    return batch


def compute_accuracy_batch(model, tokenizer, token_a_id, token_b_id, batch, use_demo=True):
    """
    Compute accuracy for a batch of test examples with optional demonstration.

    For each test example, we compute 4 predictions:
    1. Demo noflip, Test A=yw B=yl -> should prefer A
    2. Demo noflip, Test A=yl B=yw -> should prefer B
    3. Demo flip, Test A=yw B=yl -> should prefer A
    4. Demo flip, Test A=yl B=yw -> should prefer B

    Returns: (test_idx, 4 probabilities of correct answer) for each item in batch
    """
    all_prompts = []
    metadata = []

    # For each item in batch, create 4 prompts
    for item in batch:
        if use_demo:
            demo_example = {
                'prompt': item['demo_prompt'],
                'all_generated_responses': [item['demo_yw'], item['demo_yl']]
            }
        else:
            demo_example = None

        test_yw = item['test_yw']
        test_yl = item['test_yl']
        test_prompt = item['test_prompt']

        if use_demo:
            # 1. Demo A=yw [[A]], Test A=yw B=yl -> should prefer A
            all_prompts.append(create_prompt_template(
                test_prompt, test_yw, test_yl, demo_example, context_flip=False))

            # 2. Demo A=yw [[A]], Test A=yl B=yw -> should prefer B
            all_prompts.append(create_prompt_template(
                test_prompt, test_yl, test_yw, demo_example, context_flip=False))

            # 3. Demo A=yl [[B]], Test A=yw B=yl -> should prefer A
            all_prompts.append(create_prompt_template(
                test_prompt, test_yw, test_yl, demo_example, context_flip=True))

            # 4. Demo A=yl [[B]], Test A=yl B=yw -> should prefer B
            all_prompts.append(create_prompt_template(
                test_prompt, test_yl, test_yw, demo_example, context_flip=True))
        else:
            # No context - just 2 prompts (test both orderings)
            # 1. Test A=yw B=yl -> should prefer A
            all_prompts.append(create_prompt_template(
                test_prompt, test_yw, test_yl, None, context_flip=False))

            # 2. Test A=yl B=yw -> should prefer B
            all_prompts.append(create_prompt_template(
                test_prompt, test_yl, test_yw, None, context_flip=False))

            # Duplicate for consistency (4 values)
            all_prompts.append(create_prompt_template(
                test_prompt, test_yw, test_yl, None, context_flip=False))

            all_prompts.append(create_prompt_template(
                test_prompt, test_yl, test_yw, None, context_flip=False))

        metadata.append(item['test_idx'])

    # Tokenize all prompts at once
    inputs = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size * 4, seq_len, vocab_size)

        # Get last token logits for each prompt
        last_token_logits = logits[:, -1, :]  # (batch_size * 4, vocab_size)

        # Get logits for tokens A and B
        logit_a = last_token_logits[:, token_a_id]  # (batch_size * 4)
        logit_b = last_token_logits[:, token_b_id]  # (batch_size * 4)

        # Compute probabilities
        logits_ab = torch.stack([logit_a, logit_b], dim=1)  # (batch_size * 4, 2)
        probs_ab = F.softmax(logits_ab, dim=1)  # (batch_size * 4, 2)

    # Extract results
    results = []
    batch_size = len(batch)

    for i in range(batch_size):
        test_idx = metadata[i]

        # Extract probabilities for this item (4 prompts)
        start_idx = i * 4

        # For all 4 prompts: get probability of correct answer
        # Prompt 0: should prefer A, so prob_correct = P(A)
        prob_correct_0 = probs_ab[start_idx + 0, 0].item()

        # Prompt 1: should prefer B, so prob_correct = P(B)
        prob_correct_1 = probs_ab[start_idx + 1, 1].item()

        # Prompt 2: should prefer A, so prob_correct = P(A)
        prob_correct_2 = probs_ab[start_idx + 2, 0].item()

        # Prompt 3: should prefer B, so prob_correct = P(B)
        prob_correct_3 = probs_ab[start_idx + 3, 1].item()

        results.append({
            'test_idx': test_idx,
            'probs': [prob_correct_0, prob_correct_1, prob_correct_2, prob_correct_3],
        })

    return results


def compute_best_worst_demo_indices(icl_gain_results, top_k=5, rank_by='snr_prob_gain'):
    """
    Compute best and worst demonstration indices based on prediction metrics.

    At prediction time, we average over both test orderings for each context configuration.
    So for each training demo, we have num_validation × 2 data points (2 context configs).

    The gain is computed as: log P(correct|with context) - log P(correct|no context)
    where the probabilities are first averaged across test orderings, then converted back to log space.

    Args:
        icl_gain_results: numpy array of shape (num_training, num_validation, 6)
            [0]: log P(A|x, ctx) when demo: A=yw B=yl [[A]], test: A=yw B=yl
            [1]: log P(B|x, ctx) when demo: A=yw B=yl [[A]], test: A=yl B=yw
            [2]: log P(A|x, ctx) when demo: A=yl B=yw [[B]], test: A=yw B=yl
            [3]: log P(B|x, ctx) when demo: A=yl B=yw [[B]], test: A=yl B=yw
            [4]: log P(A|x) when test: A=yw B=yl (no context)
            [5]: log P(B|x) when test: A=yl B=yw (no context)
        top_k: number of best and worst samples to return
        rank_by: metric to rank by
            - 'snr_prob_gain': SNR of log probability gain
            - 'mean_prob_gain': Mean log probability gain
            - 'snr_acc_gain': SNR of accuracy gain
            - 'mean_acc_gain': Mean accuracy gain
            - 'snr_prob_gain_filtered': Filter by SNR acc gain (>0.05 for best, <-0.05 for worst),
                                        then rank by SNR of log probability gain

    Returns:
        best_demo_indices: list of top_k best demonstration indices
        worst_demo_indices: list of top_k worst demonstration indices
        metrics: dictionary with all computed metrics
    """
    num_training, num_validation, _ = icl_gain_results.shape

    # Computation steps:
    # 1. Convert log probs to probs
    # 2. Average probs across test orderings
    # 3. Convert averaged probs back to log probs
    # 4. Compute log prob gain: log P(correct|with context) - log P(correct|no context)
    # 5. Compute SNR of log prob gain

    # Convert log probs to probs
    prob_with_ctx_noflip_test1 = np.exp(icl_gain_results[:, :, 0])
    prob_with_ctx_noflip_test2 = np.exp(icl_gain_results[:, :, 1])
    prob_with_ctx_flip_test1 = np.exp(icl_gain_results[:, :, 2])
    prob_with_ctx_flip_test2 = np.exp(icl_gain_results[:, :, 3])
    prob_no_ctx_test1 = np.exp(icl_gain_results[:, :, 4])
    prob_no_ctx_test2 = np.exp(icl_gain_results[:, :, 5])

    # Average over test orderings for each context config
    avg_prob_ctx_noflip = (prob_with_ctx_noflip_test1 + prob_with_ctx_noflip_test2) / 2
    avg_prob_ctx_flip = (prob_with_ctx_flip_test1 + prob_with_ctx_flip_test2) / 2
    avg_prob_no_ctx = (prob_no_ctx_test1 + prob_no_ctx_test2) / 2

    # Convert averaged probs back to log probs
    avg_log_prob_ctx_noflip = np.log(avg_prob_ctx_noflip + 1e-10)
    avg_log_prob_ctx_flip = np.log(avg_prob_ctx_flip + 1e-10)
    avg_log_prob_no_ctx = np.log(avg_prob_no_ctx + 1e-10)

    # Stack 2 context configs: (num_training, num_validation, 2)
    avg_log_prob_with_ctx = np.stack([avg_log_prob_ctx_noflip, avg_log_prob_ctx_flip], axis=2)

    # Flatten to (num_training, num_validation * 2)
    avg_log_prob_with_ctx_flat = avg_log_prob_with_ctx.reshape(num_training, -1)
    avg_log_prob_no_ctx_flat = np.repeat(avg_log_prob_no_ctx, 2, axis=1)

    # Log probability gain: log P(correct|with context) - log P(correct|no context)
    log_prob_gain_flat = avg_log_prob_with_ctx_flat - avg_log_prob_no_ctx_flat
    mean_log_prob_gain = np.mean(log_prob_gain_flat, axis=1)
    std_log_prob_gain = np.std(log_prob_gain_flat, axis=1)
    snr_log_prob_gain = mean_log_prob_gain / (std_log_prob_gain + 1e-10)

    # Prepare averaged probs for accuracy calculation (flatten the same way as log probs)
    avg_prob_with_ctx = np.stack([avg_prob_ctx_noflip, avg_prob_ctx_flip], axis=2)
    avg_prob_with_ctx_flat = avg_prob_with_ctx.reshape(num_training, -1)
    avg_prob_no_ctx_flat = np.repeat(avg_prob_no_ctx, 2, axis=1)

    # Accuracy (prob > 0.5)
    acc_with_ctx_flat = (avg_prob_with_ctx_flat > 0.5).astype(float)
    acc_no_ctx_flat = (avg_prob_no_ctx_flat > 0.5).astype(float)

    # Accuracy gain
    acc_gain_flat = acc_with_ctx_flat - acc_no_ctx_flat
    mean_acc_gain = np.mean(acc_gain_flat, axis=1)
    std_acc_gain = np.std(acc_gain_flat, axis=1)
    snr_acc_gain = mean_acc_gain / (std_acc_gain + 1e-10)

    # Select ranking metric
    if rank_by == 'snr_prob_gain_filtered':
        # Filter by SNR of accuracy gain, then rank by SNR of log probability gain
        # For best: snr_acc_gain > 0.05
        # For worst: snr_acc_gain < -0.05
        good_mask = snr_acc_gain > 0.05
        bad_mask = snr_acc_gain < -0.05

        good_indices = np.where(good_mask)[0]
        bad_indices = np.where(bad_mask)[0]

        # Rank filtered demos by log prob SNR
        if len(good_indices) >= top_k:
            good_snr = snr_log_prob_gain[good_indices]
            best_in_filtered = np.argsort(good_snr)[-top_k:][::-1]
            best_demo_indices = good_indices[best_in_filtered]
        else:
            logger.warning(f"Only {len(good_indices)} demos with SNR acc gain > 0.05, requested {top_k}")
            if len(good_indices) > 0:
                good_snr = snr_log_prob_gain[good_indices]
                best_in_filtered = np.argsort(good_snr)[::-1]
                best_demo_indices = good_indices[best_in_filtered]
            else:
                # Fallback to top by log prob SNR if no demos pass filter
                logger.warning("No demos with SNR acc gain > 0.05, using unfiltered ranking")
                best_demo_indices = np.argsort(snr_log_prob_gain)[-top_k:][::-1]

        if len(bad_indices) >= top_k:
            bad_snr = snr_log_prob_gain[bad_indices]
            worst_in_filtered = np.argsort(bad_snr)[:top_k]
            worst_demo_indices = bad_indices[worst_in_filtered]
        else:
            logger.warning(f"Only {len(bad_indices)} demos with SNR acc gain < -0.05, requested {top_k}")
            if len(bad_indices) > 0:
                bad_snr = snr_log_prob_gain[bad_indices]
                worst_in_filtered = np.argsort(bad_snr)
                worst_demo_indices = bad_indices[worst_in_filtered]
            else:
                # Fallback to bottom by log prob SNR if no demos pass filter
                logger.warning("No demos with SNR acc gain < -0.05, using unfiltered ranking")
                worst_demo_indices = np.argsort(snr_log_prob_gain)[:top_k]

    else:
        # Standard ranking methods
        ranking_metrics = {
            'snr_prob_gain': snr_log_prob_gain,
            'mean_prob_gain': mean_log_prob_gain,
            'snr_acc_gain': snr_acc_gain,
            'mean_acc_gain': mean_acc_gain,
        }

        if rank_by not in ranking_metrics:
            raise ValueError(f"Invalid rank_by: {rank_by}. Must be one of: {list(ranking_metrics.keys())} or 'snr_prob_gain_filtered'")

        ranking_values = ranking_metrics[rank_by]

        # Find top_k best and worst
        best_demo_indices = np.argsort(ranking_values)[-top_k:][::-1]
        worst_demo_indices = np.argsort(ranking_values)[:top_k]

    metrics = {
        'mean_log_prob_gain': mean_log_prob_gain,
        'std_log_prob_gain': std_log_prob_gain,
        'snr_log_prob_gain': snr_log_prob_gain,
        'mean_acc_gain': mean_acc_gain,
        'std_acc_gain': std_acc_gain,
        'snr_acc_gain': snr_acc_gain,
        'rank_by': rank_by,
    }

    return best_demo_indices.tolist(), worst_demo_indices.tolist(), metrics


def test_with_demo(model_name, dataset_name, persona_id, demo_idx, demo_type, output_dir, batch_size=4):
    """Test accuracy on 512 test examples using a specific demonstration."""
    # Initialize accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
        if demo_type == "no_context":
            logger.info(f"Testing with no context (baseline)")
        else:
            logger.info(f"Testing with {demo_type} demonstration (idx={demo_idx})")
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Batch size per device: {batch_size}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Get token IDs
    token_a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    token_b_id = tokenizer.encode("B", add_special_tokens=False)[0]

    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    filtered_dataset, actual_persona_id = filter_dataset_by_persona(dataset, persona_id)
    persona_data = prepare_persona_dataset_as_pairwise(filtered_dataset)

    if accelerator.is_main_process:
        logger.info(f"Total samples for persona {actual_persona_id}: {len(persona_data)}")

    # Split the data
    test_set = persona_data[:512]
    validation_set = persona_data[512:562]
    training_examples = persona_data[562:]

    if accelerator.is_main_process:
        logger.info(f"Test set: {len(test_set)} examples")
        if demo_type != "no_context":
            logger.info(f"Training examples: {len(training_examples)} examples")

    # Get the demonstration (if not no_context)
    if demo_type == "no_context":
        # Create a dummy demo_example for TestDataset compatibility
        demo_example = {'prompt': '', 'all_generated_responses': ['', '']}
    else:
        demo_example = training_examples[demo_idx]
        if accelerator.is_main_process:
            logger.info(f"\n{demo_type.upper()} DEMONSTRATION:")
            logger.info(f"Prompt: {demo_example['prompt'][:200]}...")

    # Create dataset and dataloader
    test_dataset = TestDataset(test_set, demo_example)

    if accelerator.is_main_process:
        logger.info(f"Total test examples: {len(test_dataset)}")

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Prepare with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    # Process batches
    all_results = []

    if accelerator.is_main_process:
        logger.info("Starting computation...")

    for batch in tqdm(dataloader, desc="Processing", disable=not accelerator.is_main_process):
        use_demo = (demo_type != "no_context")
        batch_results = compute_accuracy_batch(model, tokenizer, token_a_id, token_b_id, batch, use_demo=use_demo)
        all_results.extend(batch_results)

    # Gather results from all processes
    if accelerator.num_processes > 1:
        gathered_results = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered_results, all_results)
        if accelerator.is_main_process:
            all_results = []
            for results_list in gathered_results:
                all_results.extend(results_list)

    # Only main process saves
    if accelerator.is_main_process:
        logger.info(f"Gathered {len(all_results)} results")

        # Reconstruct result array
        probs = np.zeros((len(test_set), 4))

        for result in all_results:
            i = result['test_idx']
            probs[i, :] = result['probs']

        # Average probabilities across 4 runs per test example
        avg_prob_per_test = probs.mean(axis=1)  # (512,)

        # Compute accuracy: 1 if avg_prob > 0.5, else 0
        accuracy_per_test = (avg_prob_per_test > 0.5).astype(float)

        # Compute overall average accuracy
        overall_avg_accuracy = accuracy_per_test.mean()

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"test_accuracy_{demo_type}_demo_{actual_persona_id}.npz")
        np.savez(output_file,
                 probs=probs,
                 avg_prob_per_test=avg_prob_per_test,
                 accuracy_per_test=accuracy_per_test,
                 overall_avg_accuracy=overall_avg_accuracy,
                 demo_idx=demo_idx if demo_type != "no_context" else -1)

        logger.info(f"\n{'='*80}")
        logger.info(f"RESULTS for {demo_type.upper()} demonstration")
        logger.info(f"{'='*80}")
        if demo_type != "no_context":
            logger.info(f"Demo index: {demo_idx}")
        logger.info(f"Overall average accuracy: {overall_avg_accuracy:.4f}")
        logger.info(f"Average probability per test example - Mean: {avg_prob_per_test.mean():.4f}, "
                   f"Std: {avg_prob_per_test.std():.4f}")
        logger.info(f"Average probability per test example - Min: {avg_prob_per_test.min():.4f}, "
                   f"Max: {avg_prob_per_test.max():.4f}")
        logger.info(f"Accuracy per test example - Mean: {accuracy_per_test.mean():.4f}, "
                   f"Std: {accuracy_per_test.std():.4f}")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test accuracy with best/worst demonstrations")
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
                       default="./icl_gain_results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--demo_type", type=str, choices=["best", "worst", "no_context", "all"],
                       default="all",
                       help="Which demonstration to test with (all includes best, worst, and no_context)")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of best and worst samples to test")
    parser.add_argument("--rank_by", type=str, default="snr_prob_gain",
                       choices=["snr_prob_gain", "mean_prob_gain", "snr_acc_gain", "mean_acc_gain", "snr_prob_gain_filtered"],
                       help="Metric to rank demonstrations by")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load ICL gain results
    logger.info(f"Loading ICL gain results from: {args.icl_gain_results_file}")
    icl_gain_results = np.load(args.icl_gain_results_file)
    logger.info(f"ICL gain results shape: {icl_gain_results.shape}")

    # Compute best and worst demonstration indices
    logger.info(f"Computing top {args.top_k} best and worst demonstration indices using {args.rank_by}...")
    best_demo_indices, worst_demo_indices, metrics = compute_best_worst_demo_indices(
        icl_gain_results, top_k=args.top_k, rank_by=args.rank_by)

    logger.info(f"\n{'='*80}")
    logger.info(f"Top {args.top_k} BEST demonstrations (ranked by {args.rank_by}):")
    logger.info(f"{'='*80}")
    for rank, idx in enumerate(best_demo_indices, 1):
        logger.info(f"  Rank {rank}: Index {idx}")
        logger.info(f"    Log prob gain: {metrics['mean_log_prob_gain'][idx]:.4f} ± {metrics['std_log_prob_gain'][idx]:.4f} (SNR: {metrics['snr_log_prob_gain'][idx]:.4f})")
        logger.info(f"    Acc gain: {metrics['mean_acc_gain'][idx]:.4f} ± {metrics['std_acc_gain'][idx]:.4f} (SNR: {metrics['snr_acc_gain'][idx]:.4f})")

    logger.info(f"\n{'='*80}")
    logger.info(f"Top {args.top_k} WORST demonstrations (ranked by {args.rank_by}):")
    logger.info(f"{'='*80}")
    for rank, idx in enumerate(worst_demo_indices, 1):
        logger.info(f"  Rank {rank}: Index {idx}")
        logger.info(f"    Log prob gain: {metrics['mean_log_prob_gain'][idx]:.4f} ± {metrics['std_log_prob_gain'][idx]:.4f} (SNR: {metrics['snr_log_prob_gain'][idx]:.4f})")
        logger.info(f"    Acc gain: {metrics['mean_acc_gain'][idx]:.4f} ± {metrics['std_acc_gain'][idx]:.4f} (SNR: {metrics['snr_acc_gain'][idx]:.4f})")

    logger.info(f"\n{'='*80}")
    logger.info(f"Overall statistics:")
    logger.info(f"{'='*80}")
    logger.info(f"Mean log prob gain: {np.mean(metrics['mean_log_prob_gain']):.4f} ± {np.std(metrics['mean_log_prob_gain']):.4f}")
    logger.info(f"Mean acc gain: {np.mean(metrics['mean_acc_gain']):.4f} ± {np.std(metrics['mean_acc_gain']):.4f}")
    logger.info(f"SNR log prob gain: {np.mean(metrics['snr_log_prob_gain']):.4f} ± {np.std(metrics['snr_log_prob_gain']):.4f}")
    logger.info(f"SNR acc gain: {np.mean(metrics['snr_acc_gain']):.4f} ± {np.std(metrics['snr_acc_gain']):.4f}\n")

    # Save the demo indices for future use
    os.makedirs(args.output_dir, exist_ok=True)
    indices_file = os.path.join(args.output_dir, f"top{args.top_k}_best_worst_demo_indices_{args.rank_by}.npz")
    np.savez(indices_file,
             best_demo_indices=best_demo_indices,
             worst_demo_indices=worst_demo_indices,
             top_k=args.top_k,
             **metrics)
    logger.info(f"Saved demo indices to: {indices_file}\n")

    # Run tests for best demonstrations
    if args.demo_type in ["best", "all"]:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with TOP {args.top_k} BEST demonstrations")
        logger.info(f"{'='*80}\n")
        for rank, demo_idx in enumerate(best_demo_indices, 1):
            logger.info(f"\n--- Testing BEST demo rank {rank} (index {demo_idx}) ---\n")
            test_with_demo(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                persona_id=args.persona_id,
                demo_idx=demo_idx,
                demo_type=f"best_rank{rank}",
                output_dir=args.output_dir,
                batch_size=args.batch_size
            )

    # Run tests for worst demonstrations
    if args.demo_type in ["worst", "all"]:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with TOP {args.top_k} WORST demonstrations")
        logger.info(f"{'='*80}\n")
        for rank, demo_idx in enumerate(worst_demo_indices, 1):
            logger.info(f"\n--- Testing WORST demo rank {rank} (index {demo_idx}) ---\n")
            test_with_demo(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                persona_id=args.persona_id,
                demo_idx=demo_idx,
                demo_type=f"worst_rank{rank}",
                output_dir=args.output_dir,
                batch_size=args.batch_size
            )

    # Run test with no context
    if args.demo_type in ["no_context", "all"]:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with NO CONTEXT (baseline)")
        logger.info(f"{'='*80}\n")
        test_with_demo(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            persona_id=args.persona_id,
            demo_idx=None,  # No demo index for no_context
            demo_type="no_context",
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
