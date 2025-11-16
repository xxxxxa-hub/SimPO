#!/usr/bin/env python3
"""
Script to test ICL gain hypothesis.

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
    """Create a prompt for pairwise comparison with optional context example.

    Args:
        question: The question to evaluate
        response_a: First response
        response_b: Second response
        context_example: Optional context example dict with 'prompt' and 'all_generated_responses'
        context_flip: If True, flip the order in context (A=yl, B=yw, label=B);
                     If False, use normal order (A=yw, B=yl, label=A)

    Returns:
        Prompt string
    """
    prompt_text = ""

    # Add context example if provided
    if context_example is not None:
        ctx_question = context_example['prompt']
        ctx_yw = context_example['all_generated_responses'][0]  # winning response
        ctx_yl = context_example['all_generated_responses'][1]  # losing response

        prompt_text += "# Example\n"
        prompt_text += f"## Question\n{ctx_question}\n\n"

        if context_flip:
            # A=yl (losing), B=yw (winning), label=B
            prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yl}\n"
            prompt_text += f"[The End of Assistant A's Answer]\n\n"
            prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yw}\n"
            prompt_text += f"[The End of Assistant B's Answer]\n\n"
            prompt_text += "## Preferred answer: [[B]]\n\n"
        else:
            # A=yw (winning), B=yl (losing), label=A
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


def get_preference_prob(model, tokenizer, token_a_id, token_b_id,
                       question, response_a, response_b,
                       context_example=None, context_flip=False, target_token='A'):
    """Get the probability of preferring a specific response.

    Args:
        model: The language model
        tokenizer: The tokenizer
        token_a_id: Token ID for "A"
        token_b_id: Token ID for "B"
        question: The question
        response_a: First response (shown as A)
        response_b: Second response (shown as B)
        context_example: Optional context example
        context_flip: Whether to flip the context demonstration order
        target_token: Which token probability to return ('A' or 'B')

    Returns:
        Probability of the target token
    """
    prompt_text = create_prompt_template(question, response_a, response_b, context_example, context_flip)

    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get logits for the last token
        last_token_logits = logits[0, -1, :]

        # Get logits for tokens "A" and "B"
        logit_a = last_token_logits[token_a_id].item()
        logit_b = last_token_logits[token_b_id].item()

        # Compute probabilities using softmax
        logits_ab = torch.tensor([logit_a, logit_b])
        probs_ab = F.softmax(logits_ab, dim=0)

        if target_token == 'A':
            return probs_ab[0].item()
        else:
            return probs_ab[1].item()


def test_hypothesis(model_name, dataset_name, persona_id, output_dir, device="cuda"):
    """Run the hypothesis test.

    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset
        persona_id: Persona ID to filter (None for auto-detect)
        output_dir: Directory to save results
        device: Device to run on
    """
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Get token IDs for A and B
    token_a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    token_b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    logger.info(f"Token A ID: {token_a_id}, Token B ID: {token_b_id}")

    # Load dataset and filter to one persona
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    filtered_dataset, actual_persona_id = filter_dataset_by_persona(dataset, persona_id)
    persona_data = prepare_persona_dataset_as_pairwise(filtered_dataset)

    logger.info(f"Total samples for persona {actual_persona_id}: {len(persona_data)}")

    # Split the data
    test_set = persona_data[:512]  # First 512 rows
    validation_set = persona_data[512:562]  # Next 50 rows
    training_examples = persona_data[562:]  # Remaining examples

    logger.info(f"Test set: {len(test_set)} examples")
    logger.info(f"Validation set: {len(validation_set)} examples")
    logger.info(f"Training examples: {len(training_examples)} examples")

    # Initialize result array
    # Shape: (num_training_examples, num_validation_examples, 6)
    # 6 values: 4 with context (2 demo arrangements × 2 test arrangements) + 2 without context
    num_training = len(training_examples)
    num_validation = len(validation_set)
    results = np.zeros((num_training, num_validation, 6))

    logger.info(f"Result array shape: {results.shape}")
    logger.info("Starting computation...")

    # For each training example as context
    breakpoint()
    for i, context_example in enumerate(tqdm(training_examples, desc="Training examples")):
        # For each validation example
        for j, val_example in enumerate(tqdm(validation_set, desc=f"Validation (train {i+1}/{num_training})", leave=False)):
            x = val_example['prompt']
            yw = val_example['all_generated_responses'][0]  # winning response
            yl = val_example['all_generated_responses'][1]  # losing response

            # WITH CONTEXT - Demo arrangement 1: A=yw, B=yl, label=A (context_flip=False)
            # Test arrangement 1: A=yw, B=yl (correct answer is A)
            prob_ctx_noflip_test_a = get_preference_prob(
                model, tokenizer, token_a_id, token_b_id,
                question=x, response_a=yw, response_b=yl,
                context_example=context_example, context_flip=False,
                target_token='A'
            )

            # Test arrangement 2: A=yl, B=yw (correct answer is B)
            prob_ctx_noflip_test_b = get_preference_prob(
                model, tokenizer, token_a_id, token_b_id,
                question=x, response_a=yl, response_b=yw,
                context_example=context_example, context_flip=False,
                target_token='B'
            )

            # WITH CONTEXT - Demo arrangement 2: A=yl, B=yw, label=B (context_flip=True)
            # Test arrangement 1: A=yw, B=yl (correct answer is A)
            prob_ctx_flip_test_a = get_preference_prob(
                model, tokenizer, token_a_id, token_b_id,
                question=x, response_a=yw, response_b=yl,
                context_example=context_example, context_flip=True,
                target_token='A'
            )

            # Test arrangement 2: A=yl, B=yw (correct answer is B)
            prob_ctx_flip_test_b = get_preference_prob(
                model, tokenizer, token_a_id, token_b_id,
                question=x, response_a=yl, response_b=yw,
                context_example=context_example, context_flip=True,
                target_token='B'
            )

            # WITHOUT CONTEXT
            # Test arrangement 1: A=yw, B=yl (correct answer is A)
            prob_no_ctx_test_a = get_preference_prob(
                model, tokenizer, token_a_id, token_b_id,
                question=x, response_a=yw, response_b=yl,
                context_example=None,
                target_token='A'
            )

            # Test arrangement 2: A=yl, B=yw (correct answer is B)
            prob_no_ctx_test_b = get_preference_prob(
                model, tokenizer, token_a_id, token_b_id,
                question=x, response_a=yl, response_b=yw,
                context_example=None,
                target_token='B'
            )

            # Store log probabilities
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            results[i, j, 0] = np.log(prob_ctx_noflip_test_a + eps)  # Demo A=yw, Test A=yw
            results[i, j, 1] = np.log(prob_ctx_noflip_test_b + eps)  # Demo A=yw, Test B=yw
            results[i, j, 2] = np.log(prob_ctx_flip_test_a + eps)    # Demo B=yw, Test A=yw
            results[i, j, 3] = np.log(prob_ctx_flip_test_b + eps)    # Demo B=yw, Test B=yw
            results[i, j, 4] = np.log(prob_no_ctx_test_a + eps)      # No context, Test A=yw
            results[i, j, 5] = np.log(prob_no_ctx_test_b + eps)      # No context, Test B=yw

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
    logger.info(f"\nGains can be computed as:")
    logger.info(f"  gain_demo_noflip_test_a = results[:, :, 0] - results[:, :, 4]")
    logger.info(f"  gain_demo_noflip_test_b = results[:, :, 1] - results[:, :, 5]")
    logger.info(f"  gain_demo_flip_test_a   = results[:, :, 2] - results[:, :, 4]")
    logger.info(f"  gain_demo_flip_test_b   = results[:, :, 3] - results[:, :, 5]")
    logger.info(f"{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test ICL gain hypothesis")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to use for evaluation")
    parser.add_argument("--dataset_name", type=str,
                       default="sher222/persona-iterative-responses",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--persona_id", type=str, default="cdf7cefb-7341-41d7-a193-ff0f2f962cf9",
                       help="Specific persona UUID to filter for (None for auto-detect)")
    parser.add_argument("--output_dir", type=str,
                       default="./icl_gain_results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    return parser.parse_args()


def main():
    args = parse_args()

    test_hypothesis(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        persona_id=args.persona_id,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
