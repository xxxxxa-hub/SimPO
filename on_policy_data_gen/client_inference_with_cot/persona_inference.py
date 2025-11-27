"""
Persona inference from demonstrations.

This module infers a user's persona/preferences based on their demonstration examples,
including which responses they prefer and what characteristics those preferences indicate.
"""

import json
import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def infer_persona_description(
    client,
    demo_examples: List[Dict],
    yw_first_flags: List[bool],
    reasoning_analyses: Optional[List[str]] = None,
    output_file: Optional[str] = None
) -> str:
    """
    Generate an inferred persona description based on demonstration examples.

    This function analyzes the preference patterns in demonstrations and the
    differences between responses to infer what kind of user/persona would
    prefer this pattern of responses.

    Args:
        client: Inference client to generate persona description
        demo_examples: List of demonstration examples
        yw_first_flags: List of flags indicating if ctx_yw should appear first for each demonstration
        reasoning_analyses: Optional pre-generated reasoning about differences
        output_file: Optional path to save the persona description as JSON

    Returns:
        Generated persona description as a text string
    """
    logger.info("Inferring persona description from demonstrations...")

    # Build context from demonstrations (same format as test examples)
    demo_context = ""

    for i, (demo_example, yw_first, reasoning) in enumerate(
        zip(demo_examples, yw_first_flags, reasoning_analyses or [None] * len(demo_examples))
    ):
        ctx_question = demo_example['prompt']
        ctx_yw = demo_example['all_generated_responses'][0]
        ctx_yl = demo_example['all_generated_responses'][1]

        demo_context += f"# Example {i + 1}\n"
        demo_context += f"## Question\n{ctx_question}\n\n"

        if yw_first:
            demo_context += f"## Response A\n{ctx_yw}\n\n"
            demo_context += f"## Response B\n{ctx_yl}\n\n"
        else:
            demo_context += f"## Response A\n{ctx_yl}\n\n"
            demo_context += f"## Response B\n{ctx_yw}\n\n"

        demo_context += f"## Analysis\n{reasoning}\n\n" if reasoning else "## Analysis\n\n"

        if yw_first:
            demo_context += f"## Preferred Response\n[[A]]\n\n"
        else:
            demo_context += f"## Preferred Response\n[[B]]\n\n"

    # Create prompt to infer persona
    persona_prompt = f"""Based on the user demonstration examples below where they consistently choose certain types of responses, infer what kind of persona or preferences this user has. Focus on:

1. What characteristics the user seems to value (e.g., detail level, tone, style, approach)
2. What their expertise level might be
3. What their communication preferences are
4. Any other patterns that emerge from their choices

{demo_context}

# Inferred Persona Description

Based on the examples above, this user appears to be someone who:
"""

    # Generate persona description
    persona_description = client.generate_text(persona_prompt)
    logger.info(f"Generated persona description:\n{persona_description}")

    # Save to file if output_file is provided
    if output_file:
        logger.info(f"Saving persona description to {output_file}")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({'persona_description': persona_description}, f, indent=2)
        logger.info(f"Persona description saved to: {output_file}")

    return persona_description


def create_persona_context(persona_description: str) -> str:
    """
    Format the persona description for use in prompts.

    Args:
        persona_description: The inferred persona description

    Returns:
        Formatted persona context for inclusion in prompts
    """
    return f"""## User Persona Context
{persona_description}

Please evaluate the following responses with this user's preferences in mind.
"""
