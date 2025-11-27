"""
Utilities for creating evaluation prompts with demonstrations.

This module handles:
- Creating prompt templates with k-shot demonstrations
- Formatting demonstrations with pre-generated reasonings
"""

from typing import List, Dict, Optional


def create_prompt_template(
    question: str,
    response_a: str,
    response_b: str,
    demo_examples: Optional[List[Dict]] = None,
    yw_first_flags: Optional[List[bool]] = None,
    demo_reasonings: Optional[List[str]] = None
) -> str:
    """
    Create a prompt for pairwise comparison with k demonstration contexts.

    This creates a prompt that:
    1. Shows k demonstration examples with their reasoning analysis
    2. Includes the current query to be evaluated
    3. Sets up for the model to generate analysis and verdict

    Args:
        question: The test question to evaluate
        response_a: Response A for test question
        response_b: Response B for test question
        demo_examples: List of demonstration examples
        yw_first_flags: List of flags indicating if ctx_yw should appear first for demonstrations
        demo_reasonings: List of pre-generated reasoning texts for demonstrations

    Returns:
        Formatted prompt string
    """
    prompt_text = ""

    # Add demonstrations with pre-generated reasonings
    if demo_examples is not None and yw_first_flags is not None and demo_reasonings is not None:
        for i, (demo_example, yw_first, reasoning) in enumerate(
            zip(demo_examples, yw_first_flags, demo_reasonings)
        ):
            ctx_question = demo_example['prompt']
            ctx_yw = demo_example['all_generated_responses'][0]
            ctx_yl = demo_example['all_generated_responses'][1]

            prompt_text += f"# Example {i+1}\n"
            prompt_text += f"## Question\n{ctx_question}\n\n"

            if yw_first:
                prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yw}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yl}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"
            else:
                prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yl}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yw}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"

            prompt_text += f"## Analysis\n{reasoning}\n\n"

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
