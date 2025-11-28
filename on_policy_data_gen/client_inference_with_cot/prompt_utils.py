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
    demo_reasonings: Optional[List[str]] = None,
    test_reasoning: Optional[str] = None,
    persona_description: Optional[str] = None,
    use_reasoning: bool = False
) -> str:
    """
    Create a prompt for pairwise comparison with k demonstration contexts.

    This creates a prompt that:
    - When use_reasoning=False: Shows k demonstration examples without reasoning analysis (baseline mode)
    - When use_reasoning=True: Shows k demonstration examples with reasoning analysis and persona description

    Args:
        question: The test question to evaluate
        response_a: Response A for test question
        response_b: Response B for test question
        demo_examples: List of demonstration examples
        yw_first_flags: List of flags indicating if ctx_yw should appear first for demonstrations
        demo_reasonings: List of pre-generated reasoning texts for demonstrations (used when use_reasoning=True)
        test_reasoning: Pre-generated reasoning analysis for the test example (not used in template)
        persona_description: Inferred description of user preferences from demonstrations (used when use_reasoning=True)
        use_reasoning: Whether to include reasoning analysis and persona description (default: False)

    Returns:
        Formatted prompt string
    """
    prompt_text = ""

    # Add demonstrations
    if demo_examples is not None and yw_first_flags is not None:
        for i, (demo_example, yw_first) in enumerate(zip(demo_examples, yw_first_flags)):
            ctx_question = demo_example['prompt']
            ctx_yw = demo_example['all_generated_responses'][0]
            ctx_yl = demo_example['all_generated_responses'][1]

            prompt_text += f"# Example {i+1}\n"
            prompt_text += f"## Question\n{ctx_question}\n\n"

            if yw_first:
                prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yl}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yw}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"
                # Add reasoning analysis if enabled
                if use_reasoning and demo_reasonings is not None:
                    reasoning = demo_reasonings[i]
                    prompt_text += f"## Analysis\n{reasoning}\n"
                prompt_text += "## Preferred answer: [[B]]\n\n"
            else:
                prompt_text += f"[The Start of Assistant A's Answer]\n{ctx_yw}\n"
                prompt_text += f"[The End of Assistant A's Answer]\n\n"
                prompt_text += f"[The Start of Assistant B's Answer]\n{ctx_yl}\n"
                prompt_text += f"[The End of Assistant B's Answer]\n\n"
                # Add reasoning analysis if enabled
                if use_reasoning and demo_reasonings is not None:
                    reasoning = demo_reasonings[i]
                    prompt_text += f"## Analysis\n{reasoning}\n"
                prompt_text += "## Preferred answer: [[A]]\n\n"

    # Add persona description if enabled and available
    if use_reasoning and persona_description:
        prompt_text += f"# User Description\n{persona_description}\n\n"

    # Add task header
    prompt_text += "# Task\n"
    if use_reasoning:
        # Reasoning mode: user preference-based instructions
        if demo_examples is not None and len(demo_examples) > 0:
            if persona_description:
                prompt_text += "Based on the inferred user description and the patterns shown in the examples above, identify which response is better aligned with the user's preferences.\n\n"
            else:
                prompt_text += "Based on the patterns shown in the examples above, identify which response is better aligned with the user's preferences.\n\n"
        else:
            if persona_description:
                prompt_text += "Based on the inferred user description, identify which response is better aligned with the user's preferences.\n\n"
            else:
                prompt_text += "Identify which response is better aligned with the user's preferences.\n\n"
    else:
        # Baseline mode: simple evaluation instructions
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

    prompt_text += "## Preferred answer: [["

    return prompt_text
