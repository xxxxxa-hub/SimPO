#!/usr/bin/env python3
"""
K-shot prompt evaluation using inference client API.

This script:
1. Loads candidate demonstration indices from an npz file
2. Randomly samples k examples to construct k-shot prompts
3. Tests on 512 test examples using an inference client
4. Extracts probabilities for tokens A and B
"""

import numpy as np
import logging
import argparse
import os
import random
from tqdm import tqdm
from datasets import load_dataset
from multiprocessing import Pool
from functools import partial
from demo_selection_utils import (
    filter_dataset_by_persona,
    prepare_persona_dataset_as_pairwise
)
from huggingface_hub import InferenceClient as HFInferenceClient
from openai import OpenAI
import json
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_demonstrations(client, demo_examples: List[Dict], demo_flips: List[bool]):
    """
    Preprocess demonstrations by generating reasoning/analysis for each.

    Args:
        client: Inference client to generate reasoning
        demo_examples: List of demonstration examples
        demo_flips: List of flip flags for each demonstration

    Returns:
        List of generated reasoning texts (one per demonstration)
    """
    logger.info("Preprocessing demonstrations: generating reasoning for each example...")
    reasonings = []

    for i, (demo_example, demo_flip) in enumerate(tqdm(zip(demo_examples, demo_flips), total=len(demo_examples), desc="Generating demo reasoning")):
        ctx_question = demo_example['prompt']
        ctx_yw = demo_example['all_generated_responses'][0]
        ctx_yl = demo_example['all_generated_responses'][1]

        # Create a prompt to generate reasoning for this demonstration
        reasoning_prompt = f"""Given the following question and two assistant responses, provide a brief analysis of which response is better and why.

## Question
{ctx_question}

"""
        if demo_flip:
            reasoning_prompt += f"""[The Start of Assistant A's Answer]
{ctx_yl}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{ctx_yw}
[The End of Assistant B's Answer]

## Analysis
"""
        else:
            reasoning_prompt += f"""[The Start of Assistant A's Answer]
{ctx_yw}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{ctx_yl}
[The End of Assistant B's Answer]

## Analysis
"""

        # Generate reasoning
        reasoning = client.generate_text(reasoning_prompt, max_new_tokens=300)
        reasonings.append(reasoning)
        logger.debug(f"Demo {i+1} reasoning: {reasoning[:100]}...")

    return reasonings


def create_prompt_template(question, response_a, response_b, demo_examples=None, demo_flips=None, demo_reasonings=None):
    """
    Create a prompt for pairwise comparison with k demonstration contexts.

    Args:
        demo_reasonings: List of pre-generated reasoning texts for demonstrations
    """
    prompt_text = ""

    # Add demonstrations with pre-generated reasonings
    if demo_examples is not None and demo_flips is not None and demo_reasonings is not None:
        for i, (demo_example, demo_flip, reasoning) in enumerate(zip(demo_examples, demo_flips, demo_reasonings)):
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
                prompt_text += f"## Analysis\n{reasoning}\n"
                prompt_text += "## Preferred answer: [[B]]\n\n"
            else:
                prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yw}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yl}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"
                prompt_text += f"## Analysis\n{reasoning}\n"
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

    prompt_text += "## Analysis\nLet me analyze these responses step by step:\n"

    return prompt_text


class InferenceClient:
    """Base class for inference clients."""

    def get_token_probabilities(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """
        Get probabilities for specific tokens.

        Args:
            prompt: The prompt to evaluate
            tokens: List of tokens to get probabilities for (e.g., ["A", "B"])

        Returns:
            Dictionary mapping token to log probability
        """
        raise NotImplementedError

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate text from the model.

        Args:
            prompt: The prompt to generate from
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text
        """
        raise NotImplementedError


class HuggingFaceInferenceClient(InferenceClient):
    """Client for HuggingFace Inference API."""

    def __init__(self, model_name: str, api_token: Optional[str] = None):
        self.model_name = model_name
        self.api_token = api_token or os.environ.get("HF_TOKEN")

        if not self.api_token:
            logger.warning("No HuggingFace API token provided. Set HF_TOKEN environment variable.")

        # Use the new router endpoint by passing the full URL
        # The model parameter accepts both model IDs and full URLs
        router_url = f"https://router.huggingface.co/{model_name}"
        self.client = HFInferenceClient(
            model=router_url,
            token=self.api_token
        )

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text using HuggingFace Inference API."""
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=1.0,
                return_full_text=False
            )

            if response:
                return response

            logger.warning("No text generated from HuggingFace")
            return ""

        except Exception as e:
            logger.error(f"Error generating text with HuggingFace: {e}")
            return ""

    def get_token_probabilities(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """Get token probabilities using HuggingFace Inference API."""
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
                return_full_text=False,
                details=True
            )

            # For now, return a simple probability extraction
            # Note: Full logprob extraction requires using the text-generation inference server
            # with logprobs parameter enabled
            logger.warning("HuggingFace Inference API does not directly return token logprobs.")
            logger.warning("Consider using VLLM or TGI endpoint with logprobs enabled.")
            return {token: 0.0 for token in tokens}

        except Exception as e:
            logger.error(f"Error calling HuggingFace API: {e}")
            return {token: 0.0 for token in tokens}


class VLLMInferenceClient(InferenceClient):
    """Client for VLLM inference server with OpenAI-compatible API."""

    def __init__(self, base_url: str, model_name: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"

        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key
        )

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text using VLLM server."""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=1.0,
                echo=False
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].text

            logger.warning("No text generated from VLLM")
            return ""

        except Exception as e:
            logger.error(f"Error generating text with VLLM: {e}")
            return ""

    def get_token_probabilities(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """Get token probabilities using VLLM server."""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=5,  # Return top 5 logprobs
                echo=False
            )

            # Extract logprobs for requested tokens
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.logprobs is not None:
                    top_logprobs = choice.logprobs.top_logprobs
                    if top_logprobs and len(top_logprobs) > 0:
                        # Get first token's logprobs
                        token_logprobs = top_logprobs[0]

                        # Extract probabilities for requested tokens
                        probs = {}
                        for token in tokens:
                            # Try exact match and with whitespace
                            if token in token_logprobs:
                                probs[token] = token_logprobs[token]
                            elif f" {token}" in token_logprobs:
                                probs[token] = token_logprobs[f" {token}"]
                            else:
                                # Token not in top logprobs, set to very low value
                                probs[token] = -100.0

                        return probs

            logger.warning(f"Could not extract logprobs from response")
            return {token: -100.0 for token in tokens}

        except Exception as e:
            logger.error(f"Error calling VLLM API: {e}")
            return {token: -100.0 for token in tokens}


class OpenAIInferenceClient(InferenceClient):
    """Client for OpenAI API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=1.0,
                echo=False
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].text

            logger.warning("No text generated from OpenAI")
            return ""

        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return ""

    def get_token_probabilities(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """Get token probabilities using OpenAI API."""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=20,
                echo=False
            )

            # Extract logprobs for requested tokens
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.logprobs is not None:
                    top_logprobs = choice.logprobs.top_logprobs
                    if top_logprobs and len(top_logprobs) > 0:
                        token_logprobs = top_logprobs[0]

                        probs = {}
                        for token in tokens:
                            if token in token_logprobs:
                                probs[token] = token_logprobs[token]
                            elif f" {token}" in token_logprobs:
                                probs[token] = token_logprobs[f" {token}"]
                            else:
                                probs[token] = -100.0

                        return probs
            return {token: -100.0 for token in tokens}

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {token: -100.0 for token in tokens}


def create_inference_client(args) -> InferenceClient:
    """Factory function to create inference client based on arguments."""
    if args.inference_backend == "vllm":
        return VLLMInferenceClient(args.vllm_url, args.model_name, args.api_key)
    elif args.inference_backend == "openai":
        return OpenAIInferenceClient(args.model_name, args.api_key)
    elif args.inference_backend == "huggingface":
        return HuggingFaceInferenceClient(args.model_name, args.api_key)
    else:
        raise ValueError(f"Unknown inference backend: {args.inference_backend}")


def evaluate_kshot_prompt(
    client: InferenceClient,
    test_example: Dict,
    demo_examples: List[Dict],
    demo_flips: List[bool],
    demo_reasonings: List[str],
    ordering: int
) -> Tuple[float, float]:
    """
    Evaluate a single test example with k-shot demonstrations using pre-generated reasonings.

    Args:
        client: Inference client
        test_example: Test example to evaluate
        demo_examples: List of k demonstration examples
        demo_flips: List of k boolean flip flags
        demo_reasonings: List of pre-generated reasoning texts for demonstrations
        ordering: 0 or 1, which ordering to use for test example

    Returns:
        Tuple of (log_prob_correct, probability_correct)
    """
    test_prompt = test_example['prompt']
    test_yw = test_example['all_generated_responses'][0]
    test_yl = test_example['all_generated_responses'][1]

    # Create prompt based on ordering
    if ordering == 0:
        # A=yw, B=yl, correct answer is A
        prompt = create_prompt_template(
            test_prompt, test_yw, test_yl,
            demo_examples=demo_examples,
            demo_flips=demo_flips,
            demo_reasonings=demo_reasonings
        )
        correct_token = "A"
    else:
        # A=yl, B=yw, correct answer is B
        prompt = create_prompt_template(
            test_prompt, test_yl, test_yw,
            demo_examples=demo_examples,
            demo_flips=demo_flips,
            demo_reasonings=demo_reasonings
        )
        correct_token = "B"

    # Generate reasoning for the test example, then get the final verdict
    reasoning = client.generate_text(prompt, max_new_tokens=300)
    final_prompt = prompt + reasoning + "\n## Preferred answer: [["

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

    Args:
        args_tuple: Tuple of (test_idx, test_example, client_args, sampled_demos, sampled_flips, demo_reasonings)
                    where client_args is a dict with keys: inference_backend, model_name,
                    vllm_url, api_key

    Returns:
        Tuple of (test_idx, log_probs_both_orderings, probs_both_orderings)
    """
    test_idx, test_example, client_args, sampled_demos, sampled_flips, demo_reasonings = args_tuple

    # Create client in this worker process
    if client_args['inference_backend'] == "vllm":
        client = VLLMInferenceClient(
            client_args['vllm_url'],
            client_args['model_name'],
            client_args.get('api_key')
        )
    elif client_args['inference_backend'] == "openai":
        client = OpenAIInferenceClient(
            client_args['model_name'],
            client_args.get('api_key')
        )
    elif client_args['inference_backend'] == "huggingface":
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
        log_prob, prob = evaluate_kshot_prompt(
            client, test_example, sampled_demos, sampled_flips, demo_reasonings, ordering
        )
        log_probs.append(log_prob)
        probs.append(prob)

    return test_idx, log_probs, probs


def parse_args():
    parser = argparse.ArgumentParser(description="K-shot evaluation using inference client")

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
    parser.add_argument("--seed", type=int,
                       default=42,
                       help="Random seed")
    parser.add_argument("--num_samples", type=int,
                       default=5,
                       help="Number of random k-shot samples to evaluate per test example")
    parser.add_argument("--num_workers", type=int,
                       default=1,
                       help="Number of parallel workers for multiprocessing (1 = no parallelism)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("K-shot evaluation using inference client")
    logger.info(f"Inference backend: {args.inference_backend}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"k-shot: {args.k}")
    logger.info(f"Number of samples per test example: {args.num_samples}")

    # Load npz file to get candidate demonstration indices
    logger.info(f"\nLoading candidate demonstrations from: {args.npz_file}")
    npz_data = np.load(args.npz_file)

    # Extract candidate_demo_indices
    if 'candidate_demo_indices' in npz_data:
        candidate_demo_indices = npz_data['candidate_demo_indices'].tolist()
    else:
        logger.error("'candidate_demo_indices' not found in npz file!")
        logger.info(f"Available keys: {list(npz_data.keys())}")
        return

    logger.info(f"Number of candidate demonstrations: {len(candidate_demo_indices)}")
    logger.info(f"Candidate indices: {candidate_demo_indices}")

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
    logger.info(f"Total demonstration pool: {len(demo_pool)} examples ({len(candidate_demonstrations)} candidates + {len(validation_set)} validation)")

    # Create inference client
    logger.info(f"\nInitializing inference client...")
    client = create_inference_client(args)

    # Preprocess demonstrations: generate reasoning for each one
    logger.info(f"\nPreprocessing demonstrations (generating reasoning for {len(demo_pool)} examples)...")
    breakpoint()
    all_demo_reasonings = preprocess_demonstrations(client, demo_pool, [False] * len(demo_pool))
    logger.info(f"Successfully generated reasonings for all demonstrations")

    # Evaluate on test set
    logger.info(f"\nEvaluating on test set ({len(test_set)} examples)...")

    # Store results: (num_samples, num_test, 2 orderings)
    all_log_probs = np.zeros((args.num_samples, len(test_set), 2))
    all_probs = np.zeros((args.num_samples, len(test_set), 2))
    all_demo_indices = []
    all_demo_flips = []

    for sample_idx in range(args.num_samples):
        logger.info(f"\nSample {sample_idx + 1}/{args.num_samples}")

        # Randomly sample k demonstrations from demo_pool (same for all test examples in this sample)
        sampled_indices = random.sample(range(len(demo_pool)), args.k)
        sampled_demos = [demo_pool[i] for i in sampled_indices]
        sampled_demo_reasonings = [all_demo_reasonings[i] for i in sampled_indices]

        # Randomly sample flip flags for each demo with balanced labels
        # Explicitly ensure balanced labels: exactly k//2 True and k//2 False
        num_true = args.k // 2
        num_false = args.k - num_true
        sampled_flips = [True] * num_true + [False] * num_false
        random.shuffle(sampled_flips)

        all_demo_indices.append(sampled_indices)
        all_demo_flips.append(sampled_flips)

        logger.info(f"Sampled demo pool indices: {sampled_indices}")
        logger.info(f"Sampled demo flips: {sampled_flips}")

        # Evaluate each test example
        if args.num_workers > 1:
            # Parallel processing
            logger.info(f"Using {args.num_workers} workers for parallel processing")

            # Prepare client arguments dictionary (avoids pickling client object)
            client_args = {
                'inference_backend': args.inference_backend,
                'model_name': args.model_name,
                'vllm_url': args.vllm_url if hasattr(args, 'vllm_url') else None,
                'api_key': args.api_key if hasattr(args, 'api_key') else None
            }

            # Prepare arguments for worker function
            worker_args = [
                (test_idx, test_example, client_args, sampled_demos, sampled_flips, sampled_demo_reasonings)
                for test_idx, test_example in enumerate(test_set)
            ]

            # Use multiprocessing pool
            with Pool(processes=args.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(evaluate_test_example_worker, worker_args),
                    total=len(test_set),
                    desc=f"Sample {sample_idx + 1}"
                ))

            # Store results
            for test_idx, log_probs, probs in results:
                all_log_probs[sample_idx, test_idx, 0] = log_probs[0]
                all_log_probs[sample_idx, test_idx, 1] = log_probs[1]
                all_probs[sample_idx, test_idx, 0] = probs[0]
                all_probs[sample_idx, test_idx, 1] = probs[1]
        else:
            # Sequential processing (original implementation)
            for test_idx, test_example in enumerate(tqdm(test_set, desc=f"Sample {sample_idx + 1}")):
                # Evaluate both orderings
                for ordering in [0, 1]:
                    log_prob, prob = evaluate_kshot_prompt(
                        client, test_example, sampled_demos, sampled_flips, sampled_demo_reasonings, ordering
                    )
                    all_log_probs[sample_idx, test_idx, ordering] = log_prob
                    all_probs[sample_idx, test_idx, ordering] = prob

    # Compute statistics
    logger.info("\nComputing statistics...")

    # Average over orderings for each sample
    avg_probs_per_sample = all_probs.mean(axis=2)  # (num_samples, num_test)

    # Average over samples
    avg_probs = avg_probs_per_sample.mean(axis=0)  # (num_test,)
    std_probs = avg_probs_per_sample.std(axis=0)   # (num_test,)

    # Compute accuracy (prob > 0.5)
    accuracy_per_sample = (avg_probs_per_sample > 0.5).astype(float)
    avg_accuracy = accuracy_per_sample.mean(axis=0)

    # Overall statistics
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
        all_demo_flips=all_demo_flips,
        avg_probs=avg_probs,
        std_probs=std_probs,
        avg_accuracy=avg_accuracy,
        candidate_demo_indices=candidate_demo_indices,
        demo_pool_size=len(demo_pool),
        num_candidate_demos=len(candidate_demonstrations),
        num_validation_demos=len(validation_set),
        k=args.k,
        num_samples=args.num_samples,
        persona_id=actual_persona_id,
        model_name=args.model_name,
        inference_backend=args.inference_backend
    )

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
