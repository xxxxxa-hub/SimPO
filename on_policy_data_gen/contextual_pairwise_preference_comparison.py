#!/usr/bin/env python3
"""
Script to generate pairwise preference comparisons using Accelerate for data parallel processing.
Based on pairwise_preference_comparison.py but optimized for multi-GPU distributed inference.

For each prompt with 5 responses, generates all pairwise comparisons and computes preference probabilities
using logits of "A" and "B" tokens to create a 5x5 preference matrix.
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

class PairwiseComparisonDataset(Dataset):
    """Custom dataset for handling pairwise comparisons with batching."""

    def __init__(self, dataset, tokenizer, max_length=None, few_shot_examples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length  # None means no truncation
        self.few_shot_examples = few_shot_examples or []
        self.comparison_data = []

        # Pre-generate all comparison pairs for all samples
        self._prepare_comparisons()
    
    def _prepare_comparisons(self):
        """Pre-generate all comparison pairs across all samples."""
        for sample_idx, sample in enumerate(self.dataset):
            prompt = sample["prompt"]
            all_responses = sample["all_generated_responses"]
            
            # Ensure we have exactly 5 responses
            if len(all_responses) != 5:
                logger.warning(f"Sample {sample_idx} has {len(all_responses)} responses, expected 5. Skipping.")
                continue
            
            # Generate all pairwise comparisons for this sample
            for i in range(len(all_responses)):
                for j in range(len(all_responses)):
                    if i == j:
                        continue  # Skip diagonal elements
                    
                    # Create prompt for comparing response i vs response j
                    comparison_prompt = self._create_prompt_template(prompt, all_responses[i], all_responses[j])
                    
                    self.comparison_data.append({
                        'sample_idx': sample_idx,
                        'response_i': i,
                        'response_j': j,
                        'prompt': comparison_prompt,
                        'response_a': all_responses[i],
                        'response_b': all_responses[j],
                        'original_prompt': prompt,
                        'all_responses': all_responses,
                        'original_rm_scores': sample.get("all_rm_scores", None)
                    })
    
    def _create_prompt_template(self, question, response_a, response_b):
        """Create a prompt template for pairwise comparison with few-shot examples."""

        # Build few-shot examples if available
        messages = []

        # Add system prompt
        system_prompt = (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
            "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
            "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were "
            "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
            "of the assistants. Be as objective as possible. After careful consideration, output your final verdict by strictly following this format: "
            '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better. '
            'Output **only** "[[A]]" or "[[B]]" and nothing else.'
        )

        messages.append({"role": "system", "content": system_prompt})

        # Add few-shot examples if available
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                # Build user message with question and responses
                user_content = f"""Question: {example['question']}

[The Start of Assistant A's Answer]
{example['response_a']}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{example['response_b']}
[The End of Assistant B's Answer]"""

                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": f"[[{example['label']}]]"})

        # Add the current query
        query_content = f"""Question: {question}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]"""

        messages.append({"role": "user", "content": query_content})

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Strip BOS because tokenize() in collate function introduces extra BOS
        bos = self.tokenizer.bos_token
        if prompt.startswith(bos):
            prompt = prompt[len(bos):]

        # Append "[[" to the text after applying chat template
        prompt += "[["

        return prompt
    
    def __len__(self):
        return len(self.comparison_data)
    
    def __getitem__(self, idx):
        item = self.comparison_data[idx]
        
        # No tokenization here - done in collate_fn for efficiency
        return {
            'prompt_text': item['prompt'],  # Raw text for tokenizer in collate_fn
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


def load_few_shot_examples(dataset_name, indices, random_seed=42, permute_demonstrations=True):
    """Load few-shot examples from train split based on specified indices."""
    try:
        test_dataset = load_dataset(dataset_name, split='test')

        examples = []
        for idx in indices:
            if idx >= len(test_dataset):
                logger.warning(f"Index {idx} is out of range for train dataset (size: {len(test_dataset)})")
                continue

            sample = test_dataset[idx]

            # Assume the dataset has 'prompt', 'chosen', and 'rejected' fields
            # If the format is different, we'll need to adapt this
            question = sample.get('prompt', sample.get('question', ''))
            chosen_raw = sample.get('chosen', sample.get('response_a', ''))
            rejected_raw = sample.get('rejected', sample.get('response_b', ''))

            chosen = chosen_raw[-1]['content']
            rejected = rejected_raw[-1]['content']

            # Create first example: A=chosen, B=rejected (label=A)
            examples.append({
                'question': question,
                'response_a': chosen,
                'response_b': rejected,
                'label': 'A'
            })

            # Create second example only if permute_demonstrations is True
            if permute_demonstrations:
                examples.append({
                    'question': question,
                    'response_a': rejected,
                    'response_b': chosen,
                    'label': 'B'
                })

        # Shuffle the examples to avoid order bias
        random.seed(random_seed)
        random.shuffle(examples)

        return examples

    except Exception as e:
        logger.warning(f"Failed to load few-shot examples: {e}")
        return []


def load_few_shot_examples_from_bt_scores(bt_scores_file, indices, random_seed=42, permute_demonstrations=True):
    """Load few-shot examples from Bradley-Terry scores file.

    Selects the response with highest BT score as chosen and lowest BT score as rejected.

    Args:
        bt_scores_file: Path to JSON file with Bradley-Terry scores
        indices: List of sample IDs to use for few-shot examples
        random_seed: Random seed for shuffling
        permute_demonstrations: Whether to create permuted demonstrations
    """
    try:
        logger.info(f"Loading few-shot examples from Bradley-Terry scores file: {bt_scores_file}")
        with open(bt_scores_file, 'r') as f:
            bt_data = json.load(f)

        examples = []
        for idx in indices:
            # Find the sample with matching sample_id
            sample = None
            for item in bt_data:
                if item.get('sample_id') == idx:
                    sample = item
                    break

            if sample is None:
                logger.warning(f"Sample ID {idx} not found in Bradley-Terry scores file")
                continue

            question = sample.get('prompt', '')
            responses = sample.get('responses', [])
            bt_scores = sample.get('bt_scores', [])
            rankings = sample.get('rankings', [])

            if len(responses) == 0 or len(bt_scores) == 0:
                logger.warning(f"Sample ID {idx} has no responses or scores")
                continue

            # Get best (highest BT score) and worst (lowest BT score) responses
            best_idx = rankings[0]  # First in rankings = highest score
            worst_idx = rankings[-1]  # Last in rankings = lowest score

            chosen = responses[best_idx]
            rejected = responses[worst_idx]

            # Create first example: A=chosen, B=rejected (label=A)
            examples.append({
                'question': question,
                'response_a': chosen,
                'response_b': rejected,
                'label': 'A'
            })

            # Create second example only if permute_demonstrations is True
            if permute_demonstrations:
                examples.append({
                    'question': question,
                    'response_a': rejected,
                    'response_b': chosen,
                    'label': 'B'
                })

        # Shuffle the examples to avoid order bias
        random.seed(random_seed)
        random.shuffle(examples)

        logger.info(f"Loaded {len(examples)} few-shot examples from Bradley-Terry scores")
        return examples

    except Exception as e:
        logger.warning(f"Failed to load few-shot examples from BT scores: {e}")
        return []

def parse_args():
    parser = argparse.ArgumentParser(description="Generate pairwise preference comparisons with Accelerate")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help="Model to use for preference evaluation")
    parser.add_argument("--dataset_name", type=str, default="Alligator123/gemma2-ultrafeedback-armorm-false_qa",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--split", type=str, default="train",
                       help="Split of the dataset to process")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device for processing comparisons")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--use_compilation", action="store_true", default=False,
                       help="Whether to use torch compilation for speedup")
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                       help="Whether to use Flash Attention 2 if available")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum sequence length for tokenization (None for no truncation)")
    parser.add_argument("--few_shot_indices", type=str, default="30,3,37,18",
                       help="Comma-separated indices of train examples to use for few-shot (e.g., '30,3,37,18')")
    parser.add_argument("--few_shot_random_seed", type=int, default=42,
                       help="Random seed for shuffling few-shot examples")
    parser.add_argument("--permute_demonstrations", action="store_true", default=False,
                       help="Whether to create permuted demonstrations by swapping response order")
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
            'logit_a': result['logit_a'],
            'logit_b': result['logit_b'],
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

def main():
    # Initialize accelerator with DDP settings
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    args = parse_args()
    
    # Only log on main process
    if accelerator.is_main_process:
        logger.info(f"Loading model: {args.model_name}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Device: {accelerator.device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set pad token if not set and configure for left padding (required for generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for causal language models (critical for generation)
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
    
    # Apply optimizations
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
    
    # Load few-shot examples if specified
    few_shot_examples = []
    if args.few_shot_indices:
        try:
            indices = [int(x.strip()) for x in args.few_shot_indices.split(',')]
        except:
            indices = []
        if accelerator.is_main_process:
            logger.info(f"Loading few-shot examples from train split with indices: {indices}")
        # few_shot_examples = load_few_shot_examples(args.dataset_name, indices, args.few_shot_random_seed, args.permute_demonstrations)
        few_shot_examples = load_few_shot_examples_from_bt_scores(bt_scores_file="gpt4_pairwise_preferences_test_0shot_cot_20trials_bt_scores.json", indices=indices, random_seed=args.few_shot_random_seed, permute_demonstrations=args.permute_demonstrations)
        if accelerator.is_main_process:
            logger.info(f"Loaded {len(few_shot_examples)} few-shot examples")

    # Load dataset
    if accelerator.is_main_process:
        logger.info(f"Loading dataset: {args.dataset_name}")

    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if accelerator.is_main_process:
        logger.info(f"Processing {len(dataset)} samples")

    # Create custom dataset and dataloader
    comparison_dataset = PairwiseComparisonDataset(dataset, tokenizer, args.max_length, few_shot_examples)
    
    # Create collate function with tokenizer and max_length (precomputed kwargs)
    collate = make_collate_fn(tokenizer, args.max_length)
    
    dataloader = DataLoader(
        comparison_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle to maintain order
        collate_fn=collate,  # Efficient collate function (no per-batch kwargs rebuild)
        num_workers=0  # Avoid multiprocessing issues with tokenizers
    )
    
    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    
    if accelerator.is_main_process:
        logger.info(f"Total comparisons: {len(comparison_dataset)}")
        logger.info(f"Batches per process: {len(dataloader)}")
    
    # Process batches
    all_results = []
    start_time = time.time()
    
    with accelerator.main_process_first():
        progress_bar = tqdm.tqdm(
            dataloader, 
            desc="Processing comparisons",
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
                    'original_rm_scores': metadata['original_rm_scores'],
                    'prob_a_over_b': prob_a_over_b,
                    'logit_a': logit_a,
                    'logit_b': logit_b
                }
                all_results.append(result)
            
            # Log progress periodically
            if accelerator.is_main_process and (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (len(dataloader) - batch_idx - 1) / 60
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches | "
                          f"Avg: {avg_time:.2f}s/batch | ETA: {eta:.1f}min")
                
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    
    # Gather all results from all processes
    # if accelerator.is_main_process:
    #     logger.info("Gathering results from all processes...")
    
    # gathered_results = gather_object(all_results)

    rank = accelerator.process_index
    permute_suffix = "permuted" if args.permute_demonstrations else "no_permute"
    part_file = f"pairwise_preferences_{args.split}_{len(indices)}_shot_{permute_suffix}_rank_{rank}.json"

    with open(part_file, "w") as f:
        for item in all_results:  # make sure items are JSON-serializable (lists/dicts/numbers/strings)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    accelerator.wait_for_everyone()
    
    # Only main process handles final processing and saving
    if accelerator.is_main_process:
        # logger.info(f"Gathered {len(gathered_results)} comparison results")
        gathered_results = []
        for r in range(accelerator.num_processes):
            with open(f"pairwise_preferences_{args.split}_{len(indices)}_shot_{permute_suffix}_rank_{r}.json") as f:
                for line in f:
                    gathered_results.append(json.loads(line))

        with open(f"pairwise_preferences_{args.split}_{len(indices)}_shot_{permute_suffix}_gathered.json", "w") as f:
            for item in gathered_results:
                f.write(json.dumps(item) + "\n")
                
        # Reconstruct preference matrices
        final_results = reconstruct_preference_matrices(gathered_results)
        
        # Save results
        permute_suffix = "permuted" if args.permute_demonstrations else "no_permute"
        output_file = f"pairwise_preferences_{args.split}_{len(indices)}_shot_{permute_suffix}_temp.json"
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Processing complete. Results saved to {output_file}")
        logger.info(f"Successfully processed {len(final_results)} samples")
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time / 60:.1f} minutes")

if __name__ == "__main__":
    main()
