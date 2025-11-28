#!/usr/bin/env python3
"""
K-shot prompt evaluation using inference client API with Chain-of-Thought reasoning.

This script:
1. Loads candidate demonstration indices from an npz file
2. Preprocesses demonstrations by generating difference analysis reasoning
3. Infers user persona from demonstrations
4. Randomly samples k examples to construct k-shot prompts
5. Tests on 512 test examples using an inference client
6. Extracts probabilities for tokens A and B
"""

import numpy as np
import logging
import argparse
import os
import random
import json
from tqdm import tqdm
from datasets import load_dataset
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional

# Import from sibling modules
import sys
sys.path.insert(0, os.path.dirname(__file__))

from reasoning_utils import (
    generate_reasonings,
    load_reasonings_from_file
)
from persona_inference import infer_persona_description, create_persona_context
from inference_clients import create_inference_client
from prompt_utils import create_prompt_template

# Import from parent directory utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from demo_selection_utils import (
    filter_dataset_by_persona,
    prepare_persona_dataset_as_pairwise
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_persona_description_from_file(persona_file: str) -> str:
    """
    Load persona description from a persona output file.

    Args:
        persona_file: Path to the persona output file (JSON format)

    Returns:
        Persona description string, or None if file doesn't exist
    """
    if not persona_file or not os.path.exists(persona_file):
        return None

    try:
        with open(persona_file, 'r') as f:
            data = json.load(f)
            persona_description = data.get('persona_description')
            if persona_description:
                logger.info(f"Loaded persona description from: {persona_file}")
                return persona_description
            else:
                logger.warning(f"No persona_description field found in {persona_file}")
                return None
    except Exception as e:
        logger.warning(f"Failed to load persona description from {persona_file}: {e}")
        return None


def evaluate_kshot_prompt(
    client,
    test_example: Dict,
    demo_examples: List[Dict],
    yw_first_flags: List[bool],
    demo_reasonings: List[str],
    ordering: int,
    test_reasoning: Optional[str] = None,
    persona_description: Optional[str] = None,
    use_reasoning: bool = False
) -> Tuple[float, float]:
    """
    Evaluate a single test example with k-shot demonstrations using pre-generated reasonings.

    Args:
        client: Inference client
        test_example: Test example to evaluate
        demo_examples: List of k demonstration examples
        yw_first_flags: List of k boolean flags (True if ctx_yw should appear first)
        demo_reasonings: List of pre-generated reasoning texts for demonstrations
        ordering: 0 or 1, which ordering to use for test example
        test_reasoning: Optional pre-generated reasoning for the test example at this ordering
        persona_description: Optional inferred description of user preferences
        use_reasoning: Whether to include reasoning analysis and persona description (default: False for baseline)

    Returns:
        Tuple of (log_prob_correct, probability_correct)
    """
    test_prompt = test_example['prompt']
    test_yw = test_example['all_generated_responses'][0]
    test_yl = test_example['all_generated_responses'][1]

    # Create prompt based on ordering
    if ordering == 0:
        prompt = create_prompt_template(
            test_prompt, test_yw, test_yl,
            demo_examples=demo_examples,
            yw_first_flags=yw_first_flags,
            demo_reasonings=demo_reasonings,
            test_reasoning=test_reasoning,
            persona_description=persona_description if use_reasoning else None,
            use_reasoning=use_reasoning
        )
        correct_token = "A"
    else:
        prompt = create_prompt_template(
            test_prompt, test_yl, test_yw,
            demo_examples=demo_examples,
            yw_first_flags=yw_first_flags,
            demo_reasonings=demo_reasonings,
            test_reasoning=test_reasoning,
            persona_description=persona_description if use_reasoning else None,
            use_reasoning=use_reasoning
        )
        correct_token = "B"
    final_prompt = prompt

    # Get token probabilities for final verdict
    token_logprobs = client.get_token_probabilities(final_prompt, ["A", "B"])

    # Compute probability from logits using softmax
    logit_a = token_logprobs["A"]
    logit_b = token_logprobs["B"]

    # Log-sum-exp trick for numerical stability
    max_logit = max(logit_a, logit_b)
    exp_a = np.exp(logit_a - max_logit)
    exp_b = np.exp(logit_b - max_logit)

    prob_a = exp_a / (exp_a + exp_b)
    prob_b = exp_b / (exp_a + exp_b)

    # Get probability of correct token
    if correct_token == "A":
        prob_correct = prob_a
    else:
        prob_correct = prob_b

    log_prob_correct = np.log(prob_correct + 1e-10)

    return log_prob_correct, prob_correct


def evaluate_test_example_worker(args_tuple):
    """
    Worker function for parallel processing of test examples.

    Each worker creates its own inference client to avoid pickling issues.
    """
    test_idx, test_example, client_args, sampled_demos, sampled_yw_first, sampled_demo_reasonings, test_reasonings, persona_description, use_reasoning = args_tuple

    # Create client in this worker process
    if client_args['inference_backend'] == "vllm":
        from inference_clients import VLLMInferenceClient
        client = VLLMInferenceClient(
            client_args['vllm_url'],
            client_args['model_name'],
            client_args.get('api_key')
        )
    elif client_args['inference_backend'] == "openai":
        from inference_clients import OpenAIInferenceClient
        client = OpenAIInferenceClient(
            client_args['model_name'],
            client_args.get('api_key')
        )
    elif client_args['inference_backend'] == "huggingface":
        from inference_clients import HuggingFaceInferenceClient
        client = HuggingFaceInferenceClient(
            client_args['model_name'],
            client_args.get('api_key')
        )
    else:
        raise ValueError(f"Unknown inference backend: {client_args['inference_backend']}")

    log_probs = []
    probs = []

    # Evaluate both orderings
    for ordering in [0, 1]:
        test_reasoning = test_reasonings[test_idx][ordering] if test_reasonings else None
        log_prob, prob = evaluate_kshot_prompt(
            client, test_example, sampled_demos, sampled_yw_first, sampled_demo_reasonings, ordering,
            test_reasoning=test_reasoning,
            persona_description=persona_description,
            use_reasoning=use_reasoning
        )
        log_probs.append(log_prob)
        probs.append(prob)

    return test_idx, log_probs, probs


def parse_args():
    parser = argparse.ArgumentParser(description="K-shot evaluation using inference client with CoT reasoning")

    # Model and inference settings
    parser.add_argument("--inference_backend", type=str,
                       choices=["vllm", "openai", "huggingface"],
                       default="vllm",
                       help="Inference backend to use")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name")
    parser.add_argument("--vllm_url", type=str,
                       default="http://localhost:8000",
                       help="VLLM server URL (for vllm backend)")
    parser.add_argument("--api_key", type=str,
                       default=None,
                       help="API key (if required)")

    # Dataset settings
    parser.add_argument("--dataset_name", type=str,
                       default="sher222/persona-iterative-responses",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--persona_id", type=str,
                       default="cdf7cefb-7341-41d7-a193-ff0f2f962cf9",
                       help="Specific persona UUID to filter for")

    # Candidate demonstrations
    parser.add_argument("--npz_file", type=str,
                       required=True,
                       help="Path to .npz file containing candidate_demo_indices")
    parser.add_argument("--k", type=int,
                       default=4,
                       help="Number of demonstrations to sample")

    # Output settings
    parser.add_argument("--output_dir", type=str,
                       default="./kshot_inference_results",
                       help="Directory to save results")
    parser.add_argument("--demo_reasoning_output_file", type=str,
                       default=None,
                       help="Optional file to save generated demonstration reasoning analysis (JSON)")
    parser.add_argument("--persona_output_file", type=str,
                       default=None,
                       help="Optional file to save inferred persona description")
    parser.add_argument("--test_reasoning_output_file", type=str,
                       default=None,
                       help="Optional file to save test example reasoning analysis (JSON)")
    parser.add_argument("--seed", type=int,
                       default=42,
                       help="Random seed")
    parser.add_argument("--num_workers", type=int,
                       default=1,
                       help="Number of parallel workers for multiprocessing (1 = no parallelism)")
    parser.add_argument("--use_reasoning", action="store_true",
                       help="Enable reasoning analysis and persona description in prompts (default: False for baseline)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("K-shot evaluation using inference client with Chain-of-Thought reasoning")
    logger.info(f"Inference backend: {args.inference_backend}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"k-shot: {args.k}")
    logger.info(f"Prompt mode: {'Reasoning (with analysis & persona)' if args.use_reasoning else 'Baseline (simple evaluation)'}")
    logger.info(f"Random seed: {args.seed}")

    # Load npz file to get candidate demonstration indices
    logger.info(f"\nLoading candidate demonstrations from: {args.npz_file}")
    npz_data = np.load(args.npz_file)

    if 'candidate_demo_indices' in npz_data:
        candidate_demo_indices = npz_data['candidate_demo_indices'].tolist()
    else:
        logger.error("'candidate_demo_indices' not found in npz file!")
        logger.info(f"Available keys: {list(npz_data.keys())}")
        return

    logger.info(f"Number of candidate demonstrations: {len(candidate_demo_indices)}")

    # Load dataset
    logger.info(f"\nLoading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    filtered_dataset, actual_persona_id = filter_dataset_by_persona(dataset, args.persona_id)
    persona_data = prepare_persona_dataset_as_pairwise(filtered_dataset)

    # Split data: 512 test, 50 validation, rest for training
    test_set = persona_data[:512]
    validation_set = persona_data[512:562]
    training_examples = persona_data[562:]

    logger.info(f"Test set: {len(test_set)} examples")
    logger.info(f"Validation set: {len(validation_set)} examples")
    logger.info(f"Training examples: {len(training_examples)} examples")

    # Get candidate demonstrations from training examples
    candidate_demonstrations = [training_examples[idx] for idx in candidate_demo_indices]
    logger.info(f"Loaded {len(candidate_demonstrations)} candidate demonstrations")

    # Combine candidate demonstrations with validation set for sampling pool
    demo_pool = candidate_demonstrations + validation_set
    logger.info(f"Total demonstration pool: {len(demo_pool)} examples")

    # Sample demonstrations once (with seed fixed for reproducibility)
    logger.info(f"\nSampling {args.k} demonstrations for preprocessing and persona inference...")
    sampled_indices = random.sample(range(len(demo_pool)), args.k)
    sampled_demos = [demo_pool[i] for i in sampled_indices]

    # Create yw_first flags with balanced labels
    num_true = args.k // 2
    num_false = args.k - num_true
    sampled_yw_first = [True] * num_true + [False] * num_false
    random.shuffle(sampled_yw_first)

    logger.info(f"Sampled demonstration indices: {sampled_indices}")
    logger.info(f"yw_first flags: {sampled_yw_first}")

    # Try to load demo reasonings from demo_reasoning_output_file if available
    demo_reasonings = None
    if args.demo_reasoning_output_file:
        all_reasonings = load_reasonings_from_file(args.demo_reasoning_output_file)
        # Convert from List[List[str]] to List[str] by taking the first reasoning
        demo_reasonings = [reasonings[0] for reasonings in all_reasonings] if all_reasonings else None

    # If not loaded from file, generate via API
    if demo_reasonings is None:
        # Create inference client
        logger.info(f"\nInitializing inference client...")
        client = create_inference_client(args)
        # Preprocess sampled demonstrations: generate reasoning for each one
        logger.info(f"\nPreprocessing demonstrations (generating reasoning for {len(sampled_demos)} examples)...")
        all_demo_reasonings = generate_reasonings(
            client, sampled_demos,
            output_file=args.demo_reasoning_output_file,
            num_workers=args.num_workers,
            inference_backend=args.inference_backend,
            model_name=args.model_name,
            vllm_url=args.vllm_url if hasattr(args, 'vllm_url') else None,
            api_key=args.api_key if hasattr(args, 'api_key') else None,
            yw_first_flags=sampled_yw_first
        )
        # Extract just the reasoning string from each result (demos have one reasoning per example)
        demo_reasonings = [reasonings[0] for reasonings in all_demo_reasonings]
        logger.info(f"Successfully generated reasonings for all demonstrations")
    else:
        logger.info(f"Using cached demo reasonings from file, skipping API calls")
        # Create inference client for persona inference and evaluation
        logger.info(f"\nInitializing inference client...")
        client = create_inference_client(args)

    # Infer persona description
    if args.persona_output_file:
        # Try to load persona description from file if available
        persona_description = load_persona_description_from_file(args.persona_output_file)

        # If not loaded from file, generate via API
        if persona_description is None:
            logger.info(f"\nInferring persona description from demonstration patterns...")
            persona_description = infer_persona_description(
                client, sampled_demos, sampled_yw_first, demo_reasonings,
                output_file=args.persona_output_file
            )
        else:
            logger.info(f"Using cached persona description from file, skipping inference")

    # Generate test reasonings
    test_reasonings = None
    if args.test_reasoning_output_file:
        test_reasonings = load_reasonings_from_file(args.test_reasoning_output_file)

    # If not loaded from file, generate via API
    if test_reasonings is None:
        logger.info(f"\nGenerating reasonings for test examples...")
        test_reasonings = generate_reasonings(
            client, test_set,
            output_file=args.test_reasoning_output_file,
            num_workers=args.num_workers,
            inference_backend=args.inference_backend,
            model_name=args.model_name,
            vllm_url=args.vllm_url if hasattr(args, 'vllm_url') else None,
            api_key=args.api_key if hasattr(args, 'api_key') else None
        )
        logger.info(f"Successfully generated reasonings for all test examples")
    else:
        logger.info(f"Using cached test reasonings from file, skipping generation")

    # Evaluate on test set
    logger.info(f"\nEvaluating on test set ({len(test_set)} examples)...")
    logger.info(f"Using sampled demo pool indices: {sampled_indices}")

    # Store results
    all_log_probs = np.zeros((len(test_set), 2))
    all_probs = np.zeros((len(test_set), 2))

    # Store the sampled demo indices and yw_first flags used
    all_demo_indices = np.array(sampled_indices)
    all_yw_first_flags = np.array(sampled_yw_first)

    # Use the cached demo reasonings for evaluation
    sampled_demo_reasonings = demo_reasonings

    # Evaluate each test example
    if args.num_workers > 1:
        logger.info(f"Using {args.num_workers} workers for parallel processing")

        client_args = {
            'inference_backend': args.inference_backend,
            'model_name': args.model_name,
            'vllm_url': args.vllm_url if hasattr(args, 'vllm_url') else None,
            'api_key': args.api_key if hasattr(args, 'api_key') else None
        }

        worker_args = [
            (test_idx, test_example, client_args, sampled_demos, sampled_yw_first, sampled_demo_reasonings, test_reasonings, persona_description, args.use_reasoning)
            for test_idx, test_example in enumerate(test_set)
        ]

        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(evaluate_test_example_worker, worker_args),
                total=len(test_set),
                desc="Evaluating test examples"
            ))

        for test_idx, log_probs, probs in results:
            all_log_probs[test_idx, 0] = log_probs[0]
            all_log_probs[test_idx, 1] = log_probs[1]
            all_probs[test_idx, 0] = probs[0]
            all_probs[test_idx, 1] = probs[1]
    else:
        # Sequential processing
        for test_idx, test_example in enumerate(tqdm(test_set, desc="Evaluating test examples")):
            for ordering in [0, 1]:
                test_reasoning = test_reasonings[test_idx][ordering] if test_reasonings else None
                log_prob, prob = evaluate_kshot_prompt(
                    client, test_example, sampled_demos, sampled_yw_first, sampled_demo_reasonings, ordering,
                    test_reasoning=test_reasoning,
                    persona_description=persona_description,
                    use_reasoning=args.use_reasoning
                )
                all_log_probs[test_idx, ordering] = log_prob
                all_probs[test_idx, ordering] = prob

    # Compute statistics
    logger.info("\nComputing statistics...")

    # Average probability over the two orderings for each test example
    avg_probs = all_probs.mean(axis=1)
    std_probs = all_probs.std(axis=1)

    # Compute accuracy for each test example
    avg_accuracy = (avg_probs > 0.5).astype(float)

    mean_prob = avg_probs.mean()
    mean_accuracy = avg_accuracy.mean()

    logger.info(f"\n{'='*80}")
    logger.info(f"Results:")
    logger.info(f"{'='*80}")
    logger.info(f"Mean probability (correct): {mean_prob:.4f}")
    logger.info(f"Mean accuracy: {mean_accuracy:.4f}")
    logger.info(f"Std of probabilities across samples: {std_probs.mean():.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"kshot_results_{actual_persona_id}_k{args.k}.npz")

    np.savez(
        output_file,
        all_log_probs=all_log_probs,
        all_probs=all_probs,
        all_demo_indices=all_demo_indices,
        all_yw_first_flags=all_yw_first_flags,
        avg_probs=avg_probs,
        std_probs=std_probs,
        avg_accuracy=avg_accuracy,
        candidate_demo_indices=candidate_demo_indices,
        demo_pool_size=len(demo_pool),
        num_candidate_demos=len(candidate_demonstrations),
        num_validation_demos=len(validation_set),
        k=args.k,
        seed=args.seed,
        persona_id=actual_persona_id,
        model_name=args.model_name,
        inference_backend=args.inference_backend
    )

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
