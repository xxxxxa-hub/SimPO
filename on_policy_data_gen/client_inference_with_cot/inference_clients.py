"""
Inference client implementations for different backends.

Supports:
- VLLM (OpenAI-compatible API)
- OpenAI API
- HuggingFace Inference API
"""

import logging
import os
from typing import List, Dict, Optional

import numpy as np
from huggingface_hub import InferenceClient as HFInferenceClient
from openai import OpenAI

logger = logging.getLogger(__name__)


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
                logprobs=5,
                echo=False
            )

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

            logger.warning("Could not extract logprobs from response")
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
