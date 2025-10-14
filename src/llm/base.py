"""
Abstract base class for LLM providers.

This module defines the interface for LLM providers, enabling provider-agnostic
integration with different LLM services (Groq, OpenAI, Ollama, etc.) while
maintaining consistent behavior across the EA Assistant pipeline.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Structured response from LLM providers.
    """
    content: str
    model: str
    provider: str
    tokens_used: int
    response_time_ms: float
    finish_reason: str
    metadata: Dict[str, Any]


@dataclass
class LLMConfig:
    """
    Configuration for LLM providers.
    
    FIXED: Required parameters first, optional parameters with defaults last.
    """
    # Required parameters (no defaults)
    api_key: str
    model: str
    
    # Optional parameters (with defaults)
    provider: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout: int = 30
    retries: int = 3
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    This class defines the interface that all LLM providers must implement,
    ensuring consistent behavior across different LLM services.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM provider.

        Args:
            config: LLM configuration including API keys, model settings, etc.
        """
        self.config = config
        self.provider_name = config.provider or "unknown"
        self.model = config.model
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate provider-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt for context
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            LLMError: If generation fails
            TimeoutError: If request times out
            RateLimitError: If rate limit is exceeded
        """
        pass

    @abstractmethod
    async def batch_generate(self, prompts: List[str],
                           system_prompt: Optional[str] = None,
                           **kwargs) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts in batch.

        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt for context
            **kwargs: Provider-specific parameters

        Returns:
            List of LLMResponse objects

        Raises:
            LLMError: If batch generation fails
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Get approximate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.

        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.provider_name,
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout
        }

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry function with exponential backoff.

        Args:
            func: Function to retry
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: Last exception after all retries exhausted
        """
        for attempt in range(self.config.retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.retries - 1:
                    logger.error(f"All {self.config.retries} retries exhausted for {func.__name__}")
                    raise

                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                             f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def close(self):
        """Close any open connections (optional, override if needed)."""
        pass


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(self, message: str, provider: str, model: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.error_code = error_code


class RateLimitError(LLMError):
    """Exception raised when rate limit is exceeded."""
    pass


class TimeoutError(LLMError):
    """Exception raised when request times out."""
    pass


class AuthenticationError(LLMError):
    """Exception raised when authentication fails."""
    pass


class ModelNotFoundError(LLMError):
    """Exception raised when specified model is not available."""
    pass