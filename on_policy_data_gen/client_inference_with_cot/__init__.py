"""
Client Inference with Chain-of-Thought (CoT) Reasoning

A modular pipeline for k-shot prompt evaluation with:
- Difference analysis of demonstration pairs (not quality judgment)
- User persona inference from preferences
- Multiple inference backend support (VLLM, OpenAI, HuggingFace)
"""

from inference_clients import (
    InferenceClient,
    VLLMInferenceClient,
    OpenAIInferenceClient,
    HuggingFaceInferenceClient,
    create_inference_client
)
from demo_reasoning_utils import preprocess_demonstrations, generate_demo_reasoning
from persona_inference import infer_persona_description, create_persona_context
from prompt_utils import create_prompt_template

__all__ = [
    'InferenceClient',
    'VLLMInferenceClient',
    'OpenAIInferenceClient',
    'HuggingFaceInferenceClient',
    'create_inference_client',
    'preprocess_demonstrations',
    'generate_demo_reasoning',
    'infer_persona_description',
    'create_persona_context',
    'create_prompt_template',
]
