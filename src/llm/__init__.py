"""
LLM integration module for Alliander EA Assistant.

This module provides provider-agnostic LLM integration with support for
Groq, OpenAI, and Ollama providers, including automatic fallback and
domain-specific prompt engineering.
"""

from .base import LLMProvider, LLMResponse, LLMConfig, LLMError, RateLimitError, AuthenticationError, ModelNotFoundError
from .groq_provider import GroqProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .factory import LLMFactory, ProviderType, create_llm_provider

__all__ = [
    # Base classes
    "LLMProvider",
    "LLMResponse",
    "LLMConfig",

    # Exceptions
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",

    # Providers
    "GroqProvider",
    "OpenAIProvider",
    "OllamaProvider",

    # Factory
    "LLMFactory",
    "ProviderType",
    "create_llm_provider"
]