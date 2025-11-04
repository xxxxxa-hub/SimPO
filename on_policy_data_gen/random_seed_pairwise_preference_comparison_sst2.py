#!/usr/bin/env python3
"""
Script to study in-context learning for semantic analysis using SST2 dataset.

For each seed:
1. Randomly samples k-shot examples from the training set
2. For each query in the validation set, randomizes the order of demonstrations
3. Evaluates the model's ability to predict sentiment (positive/negative)

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
dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=60))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticAnalysisDataset(Dataset):
    """Custom dataset for in-context learning of semantic analysis with randomized demonstration order per query."""

    def __init__(self, dataset, tokenizer, max_length=None, few_shot_examples=None, query_random_seed=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.few_shot_examples = few_shot_examples or []
        self.query_random_seed = query_random_seed  # Seed for per-query randomization

        # Label mapping
        self.label_map = {0: "negative", 1: "positive"}

    def _create_prompt_template(self, sentence, query_idx):
        """Create a prompt template for semantic analysis with randomized few-shot order.

        Format: "Sentence: {sentence}. Semantic: {semantic}\n"
        """
        # Build the prompt as a single string with few-shot examples
        prompt_text = ""

        # Randomize few-shot examples order for this query
        if self.few_shot_examples:
            # Use query_idx as seed modifier for reproducibility but different order per query
            local_rng = random.Random(self.query_random_seed + query_idx)
            randomized_examples = self.few_shot_examples.copy()
            local_rng.shuffle(randomized_examples)

            for example in randomized_examples:
                prompt_text += f"Sentence: {example['sentence']}. Semantic: {example['semantic']}\n"

        # Add the current query (without the label)
        prompt_text += f"Sentence: {sentence}. Semantic: "

        return prompt_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sentence = sample['sentence'].strip()
        label = sample['label']
        semantic = self.label_map[label]

        # Create prompt with randomized demonstration order
        prompt_text = self._create_prompt_template(sentence, idx)

        return {
            'prompt_text': prompt_text,
            'sample_idx': idx,
            'sentence': sentence,
            'label': label,
            'semantic': semantic
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


def load_few_shot_examples_from_dataset(dataset, indices):
    """Load few-shot examples from the SST2 training dataset.

    Args:
        dataset: The SST2 training dataset
        indices: List of sample indices to use for few-shot examples

    Returns:
        List of few-shot examples with sentence and semantic label
    """
    label_map = {0: "negative", 1: "positive"}
    examples = []

    for idx in indices:
        if idx >= len(dataset):
            logger.warning(f"Index {idx} is out of range for dataset (size: {len(dataset)})")
            continue

        sample = dataset[idx]
        sentence = sample['sentence'].strip()
        label = sample['label']
        semantic = label_map[label]

        examples.append({
            'sentence': sentence,
            'semantic': semantic,
            'label': label
        })

    logger.info(f"Loaded {len(examples)} few-shot examples from dataset")
    return examples


def sample_few_shot_indices(train_dataset, k_shot=4, seed=42):
    """Randomly sample k indices from the training dataset.

    Args:
        train_dataset: The training dataset to sample from
        k_shot: Number of examples to sample
        seed: Random seed for reproducibility

    Returns:
        List of k randomly sampled indices
    """
    rng = random.Random(seed)
    num_samples = len(train_dataset)
    indices = rng.sample(range(num_samples), min(k_shot, num_samples))
    return sorted(indices)  # Sort for consistency in logging


def parse_args():
    parser = argparse.ArgumentParser(description="In-context learning for semantic analysis using SST2")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to use for semantic analysis")
    parser.add_argument("--dataset_name", type=str, default="stanfordnlp/sst2",
                       help="Dataset name from HuggingFace (default: stanfordnlp/sst2)")
    parser.add_argument("--train_split", type=str, default="train",
                       help="Training split for few-shot examples")
    parser.add_argument("--test_split", type=str, default="validation",
                       help="Test split for evaluation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device for processing")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (None for all)")
    parser.add_argument("--use_compilation", action="store_true", default=False,
                       help="Whether to use torch compilation for speedup")
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                       help="Whether to use Flash Attention 2 if available")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum sequence length for tokenization (None for no truncation)")
    parser.add_argument("--k_shots", type=int, nargs='+', default=[4],
                       help="Number of few-shot examples to sample (can specify multiple)")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42],
                       help="Random seeds for few-shot sampling and demonstration order (can specify multiple)")
    parser.add_argument("--output_dir", type=str, default="./sst2_icl_results",
                       help="Directory to save results")
    return parser.parse_args()

def get_sentiment_probabilities_batch_accelerate(model, batch, token_negative_id, token_positive_id, accelerator):
    """
    Compute sentiment probabilities for a batch using Accelerate.
    Returns the probability of 'negative' and 'positive' labels.
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

            # Get logits for tokens "negative" and "positive"
            logit_negative = last_token_logits[token_negative_id].item()
            logit_positive = last_token_logits[token_positive_id].item()

            # Compute probability using softmax
            logits_sentiment = torch.tensor([logit_negative, logit_positive])
            probs_sentiment = F.softmax(logits_sentiment, dim=0)
            prob_negative = probs_sentiment[0].item()
            prob_positive = probs_sentiment[1].item()

            # Predict label (0 for negative, 1 for positive)
            predicted_label = 1 if prob_positive > prob_negative else 0

            results.append({
                'prob_negative': prob_negative,
                'prob_positive': prob_positive,
                'logit_negative': logit_negative,
                'logit_positive': logit_positive,
                'predicted_label': predicted_label
            })

    return results

def aggregate_results(all_results):
    """
    Aggregate results from distributed processes and compute metrics.
    """
    # Sort by sample_idx to maintain order
    sorted_results = sorted(all_results, key=lambda x: x['sample_idx'])

    # Compute accuracy
    correct = sum(1 for r in sorted_results if r['predicted_label'] == r['true_label'])
    total = len(sorted_results)
    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    return sorted_results, accuracy

def run_experiment_for_seed(seed, k_shot, args, accelerator, model, tokenizer, token_negative_id, token_positive_id, train_dataset, test_dataset):
    """Run the semantic analysis experiment for a single seed and k_shot."""

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment with seed: {seed}, k_shot: {k_shot}")
        logger.info(f"{'='*80}")

    # Sample k-shot indices for this seed from training data
    few_shot_indices = sample_few_shot_indices(
        train_dataset=train_dataset,
        k_shot=k_shot,
        seed=seed
    )

    if accelerator.is_main_process:
        logger.info(f"Sampled few-shot indices: {few_shot_indices}")

    # Load few-shot examples using the sampled indices
    few_shot_examples = load_few_shot_examples_from_dataset(
        dataset=train_dataset,
        indices=few_shot_indices
    )

    if accelerator.is_main_process:
        logger.info(f"Loaded {len(few_shot_examples)} few-shot examples")
        for i, ex in enumerate(few_shot_examples[:3]):  # Show first 3 examples
            logger.info(f"Example {i+1}: Sentence: {ex['sentence'][:50]}... | Semantic: {ex['semantic']}")

    # Create custom dataset with randomized demonstration order per query
    evaluation_dataset = SemanticAnalysisDataset(
        test_dataset,
        tokenizer,
        args.max_length,
        few_shot_examples,
        query_random_seed=seed  # Use seed for per-query randomization
    )

    # Create collate function with tokenizer and max_length
    collate = make_collate_fn(tokenizer, args.max_length)

    dataloader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0
    )

    # Prepare dataloader with accelerator
    dataloader = accelerator.prepare(dataloader)

    if accelerator.is_main_process:
        logger.info(f"Total test samples: {len(evaluation_dataset)}")
        logger.info(f"Batches per process: {len(dataloader)}")

    # Process batches
    all_results = []
    start_time = time.time()

    with accelerator.main_process_first():
        progress_bar = tqdm.tqdm(
            dataloader,
            desc=f"Seed {seed} - Evaluating semantic analysis",
            disable=not accelerator.is_main_process
        )

    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Get sentiment probabilities for the batch
            batch_results = get_sentiment_probabilities_batch_accelerate(
                model, batch, token_negative_id, token_positive_id, accelerator
            )

            # Combine with metadata
            for i, pred_result in enumerate(batch_results):
                metadata = batch['metadata'][i]
                result = {
                    'sample_idx': metadata['sample_idx'],
                    'sentence': metadata['sentence'],
                    'true_label': metadata['label'],
                    'true_semantic': metadata['semantic'],
                    'predicted_label': pred_result['predicted_label'],
                    'predicted_semantic': 'negative' if pred_result['predicted_label'] == 0 else 'positive',
                    'prob_negative': pred_result['prob_negative'],
                    'prob_positive': pred_result['prob_positive'],
                    'logit_negative': pred_result['logit_negative'],
                    'logit_positive': pred_result['logit_positive'],
                    'correct': pred_result['predicted_label'] == metadata['label']
                }
                all_results.append(result)

            # Log progress periodically
            if accelerator.is_main_process and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (len(dataloader) - batch_idx - 1) / 60
                # Compute running accuracy
                running_correct = sum(1 for r in all_results if r['correct'])
                running_acc = running_correct / len(all_results) if all_results else 0
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches | "
                          f"Acc: {running_acc:.4f} | Avg: {avg_time:.2f}s/batch | ETA: {eta:.1f}min")

        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    # Save intermediate results per rank
    rank = accelerator.process_index
    part_file = f"{args.output_dir}/seed_{seed}_rank_{rank}.jsonl"
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
            rank_file = f"{args.output_dir}/seed_{seed}_rank_{r}.jsonl"
            with open(rank_file) as f:
                for line in f:
                    gathered_results.append(json.loads(line))

        logger.info(f"Gathered {len(gathered_results)} results for seed {seed}")

        # Aggregate and compute metrics
        final_results, accuracy = aggregate_results(gathered_results)

        # Save final results for this seed
        output_file = f"{args.output_dir}/sst2_results_seed_{seed}_k{k_shot}.json"
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        # Also save metadata about the experiment
        metadata_file = f"{args.output_dir}/metadata_seed_{seed}_k{k_shot}.json"
        metadata = {
            "seed": seed,
            "k_shot": k_shot,
            "few_shot_indices": few_shot_indices,
            "num_samples": len(final_results),
            "accuracy": accuracy,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "train_split": args.train_split,
            "test_split": args.test_split
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Seed {seed}, k_shot {k_shot} complete. Processed {len(final_results)} samples")
        logger.info(f"Final Accuracy: {accuracy:.4f}")

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
        logger.info(f"In-Context Learning for Semantic Analysis (SST2)")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Dataset: {args.dataset_name}")
        logger.info(f"Train split: {args.train_split}")
        logger.info(f"Test split: {args.test_split}")
        logger.info(f"K-shots: {args.k_shots}")
        logger.info(f"Seeds: {args.seeds}")
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

    # Pre-compute token IDs for "negative" and "positive"
    token_negative_id = tokenizer.encode("negative", add_special_tokens=False)[0]
    token_positive_id = tokenizer.encode("positive", add_special_tokens=False)[0]
    if accelerator.is_main_process:
        logger.info(f"Token 'negative' ID: {token_negative_id}, Token 'positive' ID: {token_positive_id}")

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

    # Load datasets
    if accelerator.is_main_process:
        logger.info(f"Loading dataset: {args.dataset_name}")

    train_dataset = load_dataset(args.dataset_name, split=args.train_split)
    test_dataset = load_dataset(args.dataset_name, split=args.test_split)

    if args.max_samples:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))

    if accelerator.is_main_process:
        logger.info(f"Training set size: {len(train_dataset)}")
        logger.info(f"Test set size: {len(test_dataset)}")

    # Run experiments for all combinations of k_shots and seeds
    total_experiments = len(args.k_shots) * len(args.seeds)
    experiment_num = 0

    for k_shot in args.k_shots:
        for seed in args.seeds:
            experiment_num += 1
            if accelerator.is_main_process:
                logger.info(f"\n{'='*80}")
                logger.info(f"Experiment {experiment_num}/{total_experiments}: k_shot={k_shot}, seed={seed}")
                logger.info(f"{'='*80}")

            run_experiment_for_seed(
                seed=seed,
                k_shot=k_shot,
                args=args,
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                token_negative_id=token_negative_id,
                token_positive_id=token_positive_id,
                train_dataset=train_dataset,
                test_dataset=test_dataset
            )

    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"All experiments complete!")
        logger.info(f"Ran {total_experiments} experiments with k_shots={args.k_shots} and seeds={args.seeds}")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
