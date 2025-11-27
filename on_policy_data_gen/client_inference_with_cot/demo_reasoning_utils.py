"""
Utilities for generating reasoning traces for demonstration examples.

This module handles:
- Generating difference analysis for each demonstration
- Saving reasoning traces to JSON files
"""

import json
import logging
import os
from typing import List, Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_demo_reasoning(client, demo_example: Dict, yw_first: bool) -> str:
    """
    Generate reasoning/analysis for a single demonstration by analyzing differences.

    This generates analysis that highlights differences between two responses
    from multiple perspectives (style, detail, tone, approach, accuracy, etc.).
    The analysis is for identifying personalized user preferences, not for
    general quality judgment.

    Args:
        client: Inference client to generate reasoning
        demo_example: Single demonstration example with 'prompt' and 'all_generated_responses'
        yw_first: Boolean indicating if ctx_yw should appear first (True) or second (False)

    Returns:
        Generated reasoning text analyzing differences
    """
    ctx_question = demo_example['prompt']
    ctx_yw = demo_example['all_generated_responses'][0]
    ctx_yl = demo_example['all_generated_responses'][1]

    # Create prompt to analyze differences (NOT judge which is better)
    reasoning_prompt = f"""Analyze the key differences between the two responses below across multiple perspectives (e.g., style, detail level, tone, approach, accuracy, comprehensiveness). Focus on what makes them different, not on judging which is better.

## Question
{ctx_question}

"""

    if yw_first:
        reasoning_prompt += f"""[Response A]
{ctx_yw}
[End of Response A]

[Response B]
{ctx_yl}
[End of Response B]

## Differences Analysis
"""
    else:
        reasoning_prompt += f"""[Response A]
{ctx_yl}
[End of Response A]

[Response B]
{ctx_yw}
[End of Response B]

## Differences Analysis
"""

    # Generate reasoning
    reasoning = client.generate_text(reasoning_prompt, max_new_tokens=300)
    return reasoning


def preprocess_demonstrations(
    client,
    demo_examples: List[Dict],
    yw_first_flags: List[bool],
    output_file: Optional[str] = None
) -> List[str]:
    """
    Preprocess demonstrations by generating difference analysis for each.

    This generates reasoning that analyzes the differences between two responses
    from multiple perspectives. This analysis is intended to support personalized
    preference identification, which may differ from general/objective quality judgments.

    Args:
        client: Inference client to generate reasoning
        demo_examples: List of demonstration examples
        yw_first_flags: List of flags indicating if ctx_yw should appear first for each demonstration
        output_file: Optional path to save the analysis results as JSON

    Returns:
        List of generated reasoning texts (one per demonstration)
    """
    logger.info("Preprocessing demonstrations: generating difference analysis for each example...")
    reasonings = []
    analysis_data = []

    for i, (demo_example, yw_first) in enumerate(
        tqdm(zip(demo_examples, yw_first_flags), total=len(demo_examples), desc="Generating demo analysis")
    ):
        # Generate reasoning for this demonstration
        reasoning = generate_demo_reasoning(client, demo_example, yw_first)
        reasonings.append(reasoning)
        logger.debug(f"Demo {i+1} analysis: {reasoning[:100]}...")

        # Collect data for saving
        if output_file:
            ctx_question = demo_example['prompt']
            ctx_yw = demo_example['all_generated_responses'][0]
            ctx_yl = demo_example['all_generated_responses'][1]

            analysis_data.append({
                'demo_idx': i,
                'question': ctx_question,
                'response_a': ctx_yw if yw_first else ctx_yl,
                'response_b': ctx_yl if yw_first else ctx_yw,
                'differences_analysis': reasoning
            })

    # Save analysis to JSON file if requested
    if output_file and analysis_data:
        logger.info(f"Saving difference analysis to {output_file}")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'purpose': 'Personalized user preference analysis',
                    'analysis_type': 'Response differences (not quality judgment)',
                    'num_demonstrations': len(analysis_data)
                },
                'demonstrations': analysis_data
            }, f, indent=2)
        logger.info(f"Analysis saved with {len(analysis_data)} demonstrations")

    return reasonings
