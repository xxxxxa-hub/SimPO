#!/usr/bin/env python3
"""
Script to generate pairwise preference comparisons using GPT-4o-mini API.

For 4 specified prompts with 5 responses each, generates all pairwise comparisons
and computes preference probabilities using logprobs of "A" and "B" tokens to create 5x5 preference matrices.
Uses 0-shot prompting (no few-shot examples).
"""

import json
import os
import argparse
import tqdm
import numpy as np
from datasets import load_dataset
import logging
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt_template(question, response_a, response_b):
    """Create a 0-shot prompt template for pairwise comparison with CoT."""

    # Add system prompt
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
        "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
        "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were "
        "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
        "of the assistants. Be as objective as possible. "
        "\n\nFirst, provide a brief analysis comparing both responses. "
        "Then, output your final verdict by strictly following this format: "
        '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better, or "[[TIE]]" if they are equally good. '
        "Your response must end with exactly one of: [[A]], [[B]], or [[TIE]]."
    )

    # Build the query content
    query_content = f"""Question: {question}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_content}
    ]

    return messages


def get_single_trial(client, messages, model_name, trial_num):
    """Run a single trial and return the result."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1000,
            temperature=1.0
        )
        generated_text = response.choices[0].message.content

        # Extract verdict
        if "[[A]]" in generated_text:
            verdict = 'A'
        elif "[[B]]" in generated_text:
            verdict = 'B'
        elif "[[TIE]]" in generated_text:
            verdict = 'TIE'
        else:
            logger.warning(f"Could not parse verdict from response (trial {trial_num + 1}): {generated_text[-100:]}")
            verdict = None

        return verdict, generated_text
    except Exception as e:
        logger.error(f"Error calling GPT API on trial {trial_num + 1}: {str(e)}")
        return None, None


def get_preference_with_cot(client, messages, model_name="gpt-4.1-mini", num_trials=5, max_workers=10):
    """
    Get preference using CoT reasoning by generating full responses and counting wins.
    Parallelizes API calls across trials.

    Args:
        client: OpenAI client
        messages: List of messages for the API
        model_name: Model name to use
        num_trials: Number of times to sample the model (default: 5)
        max_workers: Maximum number of parallel API calls (default: 10)

    Returns:
        tuple: (win_rate_a, num_a_wins, num_b_wins, num_ties, responses)
    """
    responses = []
    num_a_wins = 0
    num_b_wins = 0
    num_ties = 0

    # Parallelize trials
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_single_trial, client, messages, model_name, trial)
                   for trial in range(num_trials)]

        for future in as_completed(futures):
            verdict, generated_text = future.result()
            if generated_text:
                responses.append(generated_text)

            if verdict == 'A':
                num_a_wins += 1
            elif verdict == 'B':
                num_b_wins += 1
            elif verdict == 'TIE':
                num_ties += 1

    # Calculate win rate for A (ties count as 0.5)
    total_valid = num_a_wins + num_b_wins + num_ties
    if total_valid > 0:
        win_rate_a = (num_a_wins + 0.5 * num_ties) / total_valid
    else:
        win_rate_a = 0.5  # Default if no valid responses

    return win_rate_a, num_a_wins, num_b_wins, num_ties, responses


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pairwise preference comparisons using GPT-4o-mini")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini",
                       help="OpenAI model to use (default: gpt-4.1-mini)")
    parser.add_argument("--dataset_name", type=str, default="Alligator123/gemma2-ultrafeedback-armorm-false_qa",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--split", type=str, default="test",
                       help="Split of the dataset to process")
    parser.add_argument("--prompt_indices", type=str, default=None,
                       help="Comma-separated indices of prompts to evaluate (e.g., '30,3,37,18')")
    parser.add_argument("--num_random_prompts", type=int, default=None,
                       help="Number of random prompts to select from dataset (alternative to --prompt_indices)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for prompt selection (default: 42)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env variable)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                       help="Delay between API calls in seconds to respect rate limits (ignored in parallel mode)")
    parser.add_argument("--num_trials", type=int, default=20,
                       help="Number of trials to run for each comparison to estimate win rate")
    parser.add_argument("--max_workers_trials", type=int, default=10,
                       help="Maximum parallel workers for trials within each comparison (default: 10)")
    parser.add_argument("--max_workers_comparisons", type=int, default=5,
                       help="Maximum parallel workers for comparisons (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Loading dataset: {args.dataset_name}")

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)

    # Determine prompt indices
    if args.prompt_indices is not None and args.num_random_prompts is not None:
        raise ValueError("Cannot specify both --prompt_indices and --num_random_prompts. Choose one.")
    elif args.prompt_indices is not None:
        # Parse prompt indices
        prompt_indices = [int(x.strip()) for x in args.prompt_indices.split(',')]
        logger.info(f"Processing {len(prompt_indices)} prompts with indices: {prompt_indices}")
    elif args.num_random_prompts is not None:
        # Randomly select prompts
        np.random.seed(args.random_seed)
        dataset_size = len(dataset)
        if args.num_random_prompts > dataset_size:
            raise ValueError(f"--num_random_prompts ({args.num_random_prompts}) cannot be larger than dataset size ({dataset_size})")
        prompt_indices = np.random.choice(dataset_size, size=args.num_random_prompts, replace=False).tolist()
        logger.info(f"Randomly selected {len(prompt_indices)} prompts with seed {args.random_seed}: {prompt_indices}")
    else:
        raise ValueError("Must specify either --prompt_indices or --num_random_prompts")

    # Verify indices are valid
    for idx in prompt_indices:
        if idx >= len(dataset):
            raise ValueError(f"Index {idx} is out of range for dataset (size: {len(dataset)})")

    # Process each prompt
    all_results = []

    for sample_idx in prompt_indices:
        sample = dataset[sample_idx]
        prompt = sample["prompt"]
        all_responses = sample["all_generated_responses"]

        # Ensure we have exactly 5 responses
        if len(all_responses) != 5:
            logger.warning(f"Sample {sample_idx} has {len(all_responses)} responses, expected 5. Skipping.")
            continue

        logger.info(f"\nProcessing prompt {sample_idx}...")

        # Generate all pairwise comparisons (20 comparisons per prompt)
        comparisons = {}
        preference_matrix = np.full((5, 5), np.nan)

        # Create all comparison tasks
        comparison_tasks = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    comparison_tasks.append((i, j))

        progress_bar = tqdm.tqdm(total=len(comparison_tasks), desc=f"Prompt {sample_idx}")

        def process_comparison(i, j):
            """Process a single comparison."""
            messages = create_prompt_template(prompt, all_responses[i], all_responses[j])
            win_rate_a, num_a_wins, num_b_wins, num_ties, cot_responses = get_preference_with_cot(
                client, messages, args.model_name, args.num_trials, args.max_workers_trials
            )
            return i, j, win_rate_a, num_a_wins, num_b_wins, num_ties, cot_responses

        # Parallelize comparisons
        with ThreadPoolExecutor(max_workers=args.max_workers_comparisons) as executor:
            futures = [executor.submit(process_comparison, i, j) for i, j in comparison_tasks]

            for future in as_completed(futures):
                i, j, win_rate_a, num_a_wins, num_b_wins, num_ties, cot_responses = future.result()

                # Store result
                key = f"{i}_vs_{j}"
                comparisons[key] = {
                    'response_a': all_responses[i],
                    'response_b': all_responses[j],
                    'win_rate_a': win_rate_a,
                    'num_a_wins': num_a_wins,
                    'num_b_wins': num_b_wins,
                    'num_ties': num_ties,
                    'num_trials': args.num_trials,
                    'cot_responses': cot_responses
                }

                preference_matrix[i, j] = win_rate_a
                progress_bar.update(1)

        progress_bar.close()

        # Store results for this prompt
        result = {
            "sample_id": sample_idx,
            "prompt": prompt,
            "responses": all_responses,
            "preference_matrix": preference_matrix.tolist(),
            "detailed_comparisons": comparisons,
            "original_rm_scores": sample.get("all_rm_scores", None)
        }

        all_results.append(result)

    # Save results
    output_file = f"gpt4_pairwise_preferences_{args.split}_{len(prompt_indices)}shot_cot_{args.num_trials}trials.json"
    logger.info(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Processing complete. Successfully processed {len(all_results)} prompts")
    logger.info(f"Total API calls made: {len(all_results) * 20 * args.num_trials}")


if __name__ == "__main__":
    main()