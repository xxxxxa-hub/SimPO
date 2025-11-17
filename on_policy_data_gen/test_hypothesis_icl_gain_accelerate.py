#!/usr/bin/env python3
"""
Script to test ICL gain hypothesis with Accelerate for multi-GPU support.

For one user with ~700 pairwise comparisons:
- Test set: first 512 rows
- Validation set: next 50 rows (rows 512-561)
- Training examples: remaining ~200 rows (rows 562+)

For each training example i as context and each validation example (x,y):
Calculate gain_i = log P(y|x, context_i) - log P(y|x)

Due to pairwise nature, we flip both the demonstration and test examples.
We record 6 values per (training_i, validation_j) pair:

WITH CONTEXT (4 values):
1. log P(A|x, ctx) when demo: A=yw B=yl [[A]], test: A=yw B=yl
2. log P(B|x, ctx) when demo: A=yw B=yl [[A]], test: A=yl B=yw
3. log P(A|x, ctx) when demo: A=yl B=yw [[B]], test: A=yw B=yl
4. log P(B|x, ctx) when demo: A=yl B=yw [[B]], test: A=yl B=yw

NO CONTEXT (2 values):
5. log P(A|x) when test: A=yw B=yl
6. log P(B|x) when test: A=yl B=yw

Output shape: (num_training_examples, num_validation_examples, 6)
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


class ICLGainDataset(Dataset):
    """Dataset for computing ICL gains across all (training, validation) pairs."""

    def __init__(self, training_examples, validation_set):
        self.training_examples = training_examples
        self.validation_set = validation_set

        # Create all (training_idx, validation_idx) pairs
        self.pairs = []
        for i in range(len(training_examples)):
            for j in range(len(validation_set)):
                self.pairs.append((i, j))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        train_idx, val_idx = self.pairs[idx]
        context_example = self.training_examples[train_idx]
        val_example = self.validation_set[val_idx]

        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'context_prompt': context_example['prompt'],
            'context_yw': context_example['all_generated_responses'][0],
            'context_yl': context_example['all_generated_responses'][1],
            'val_prompt': val_example['prompt'],
            'val_yw': val_example['all_generated_responses'][0],
            'val_yl': val_example['all_generated_responses'][1],
        }


def collate_fn(batch):
    """Collate function to create 6 prompts per item in batch."""
    return batch  # Return as-is, we'll process in the compute function


def compute_probs_batch(model, tokenizer, token_a_id, token_b_id, batch):
    """Compute all 6 probabilities for a batch of (training, validation) pairs."""
    all_prompts = []
    metadata = []

    # For each item in batch, create 6 prompts
    for item in batch:
        context_example = {
            'prompt': item['context_prompt'],
            'all_generated_responses': [item['context_yw'], item['context_yl']]
        }

        val_yw = item['val_yw']
        val_yl = item['val_yl']
        val_prompt = item['val_prompt']

        # 1. Demo A=yw [[A]], Test A=yw B=yl
        all_prompts.append(create_prompt_template(
            val_prompt, val_yw, val_yl, context_example, context_flip=False))

        # 2. Demo A=yw [[A]], Test A=yl B=yw
        all_prompts.append(create_prompt_template(
            val_prompt, val_yl, val_yw, context_example, context_flip=False))

        # 3. Demo A=yl [[B]], Test A=yw B=yl
        all_prompts.append(create_prompt_template(
            val_prompt, val_yw, val_yl, context_example, context_flip=True))

        # 4. Demo A=yl [[B]], Test A=yl B=yw
        all_prompts.append(create_prompt_template(
            val_prompt, val_yl, val_yw, context_example, context_flip=True))

        # 5. No context, Test A=yw B=yl
        all_prompts.append(create_prompt_template(
            val_prompt, val_yw, val_yl, None, context_flip=False))

        # 6. No context, Test A=yl B=yw
        all_prompts.append(create_prompt_template(
            val_prompt, val_yl, val_yw, None, context_flip=False))

        metadata.append((item['train_idx'], item['val_idx']))

    # Tokenize all prompts at once
    inputs = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size * 6, seq_len, vocab_size)

        # Get last token logits for each prompt
        last_token_logits = logits[:, -1, :]  # (batch_size * 6, vocab_size)

        # Get logits for tokens A and B
        logit_a = last_token_logits[:, token_a_id]  # (batch_size * 6)
        logit_b = last_token_logits[:, token_b_id]  # (batch_size * 6)

        # Compute probabilities
        logits_ab = torch.stack([logit_a, logit_b], dim=1)  # (batch_size * 6, 2)
        probs_ab = F.softmax(logits_ab, dim=1)  # (batch_size * 6, 2)

    # Extract results
    results = []
    batch_size = len(batch)

    for i in range(batch_size):
        train_idx, val_idx = metadata[i]

        # Extract probabilities for this item (6 prompts)
        start_idx = i * 6

        # For prompts 0, 2, 4: we want P(A)
        # For prompts 1, 3, 5: we want P(B)
        prob_0 = probs_ab[start_idx + 0, 0].item()      # P(A) for demo noflip, test A
        prob_1 = probs_ab[start_idx + 1, 1].item()      # P(B) for demo noflip, test B
        prob_2 = probs_ab[start_idx + 2, 0].item()      # P(A) for demo flip, test A
        prob_3 = probs_ab[start_idx + 3, 1].item()      # P(B) for demo flip, test B
        prob_4 = probs_ab[start_idx + 4, 0].item()      # P(A) for no context, test A
        prob_5 = probs_ab[start_idx + 5, 1].item()      # P(B) for no context, test B

        eps = 1e-10
        results.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'log_probs': [
                np.log(prob_0 + eps),
                np.log(prob_1 + eps),
                np.log(prob_2 + eps),
                np.log(prob_3 + eps),
                np.log(prob_4 + eps),
                np.log(prob_5 + eps),
            ]
        })

    return results


def test_hypothesis(model_name, dataset_name, persona_id, output_dir, batch_size=4):
    """Run the hypothesis test with Accelerate for multi-GPU support."""
    # Initialize accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
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
        logger.info(f"Validation set: {len(validation_set)} examples")
        logger.info(f"Training examples: {len(training_examples)} examples")

    num_training = len(training_examples)
    num_validation = len(validation_set)

    # Create dataset and dataloader
    icl_dataset = ICLGainDataset(training_examples, validation_set)

    if accelerator.is_main_process:
        logger.info(f"Total pairs to process: {len(icl_dataset)}")

    dataloader = DataLoader(
        icl_dataset,
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
        batch_results = compute_probs_batch(model, tokenizer, token_a_id, token_b_id, batch)
        all_results.extend(batch_results)

    # Gather results from all processes
    if accelerator.num_processes > 1:
        # Collect results from all processes
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
        results = np.zeros((num_training, num_validation, 6))

        for result in all_results:
            i = result['train_idx']
            j = result['val_idx']
            results[i, j, :] = result['log_probs']

        # Save to .npy file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"icl_gain_results_persona_{actual_persona_id}.npy")
        np.save(output_file, results)

        logger.info(f"\n{'='*80}")
        logger.info(f"RESULTS SAVED")
        logger.info(f"{'='*80}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Shape: {results.shape}")
        logger.info(f"Description:")
        logger.info(f"  - Dimension 0: Training examples ({num_training})")
        logger.info(f"  - Dimension 1: Validation examples ({num_validation})")
        logger.info(f"  - Dimension 2: 6 values")
        logger.info(f"    [0]: log P(A|x, ctx) when demo: A=yw B=yl [[A]], test: A=yw B=yl")
        logger.info(f"    [1]: log P(B|x, ctx) when demo: A=yw B=yl [[A]], test: A=yl B=yw")
        logger.info(f"    [2]: log P(A|x, ctx) when demo: A=yl B=yw [[B]], test: A=yw B=yl")
        logger.info(f"    [3]: log P(B|x, ctx) when demo: A=yl B=yw [[B]], test: A=yl B=yw")
        logger.info(f"    [4]: log P(A|x) when test: A=yw B=yl (no context)")
        logger.info(f"    [5]: log P(B|x) when test: A=yl B=yw (no context)")
        logger.info(f"{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test ICL gain hypothesis with Accelerate")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to use for evaluation")
    parser.add_argument("--dataset_name", type=str,
                       default="sher222/persona-iterative-responses",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--persona_id", type=str, default="cdf7cefb-7341-41d7-a193-ff0f2f962cf9",
                       help="Specific persona UUID to filter for")
    parser.add_argument("--output_dir", type=str,
                       default="./icl_gain_results",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    return parser.parse_args()


def main():
    args = parse_args()

    test_hypothesis(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        persona_id=args.persona_id,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
