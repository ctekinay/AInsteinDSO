"""
OpenAI LLM provider implementation.

OpenAI provider for GPT models as a fallback option when Groq is unavailable.
Implements the LLMProvider interface for OpenAI's API.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .base import LLMProvider, LLMResponse, LLMConfig, LLMError, RateLimitError, AuthenticationError, ModelNotFoundError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider for GPT models.

    Supports models including:
    - gpt-5 (RECOMMENDED - unified system with built-in reasoning)
    - gpt-5-pro (extended reasoning for complex tasks)
    - gpt-5-thinking (deeper reasoning for harder problems)
    - gpt-4.5 (Orion - transitional model)
    - gpt-4o (multimodal capabilities)
    - gpt-3.5-turbo (legacy, fast)
    """

    BASE_URL = "https://api.openai.com/v1"

    SUPPORTED_MODELS = {
        # GPT-5 Series (August 2025 - Latest) - RECOMMENDED
        "gpt-5": {"max_tokens": 16384, "context_window": 128000},
        "gpt-5-instant": {"max_tokens": 16384, "context_window": 128000},
        "gpt-5-thinking": {"max_tokens": 16384, "context_window": 128000},
        "gpt-5-pro": {"max_tokens": 32768, "context_window": 128000},
        
        # GPT-4.5 Series (Orion - Transitional)
        "gpt-4.5": {"max_tokens": 8192, "context_window": 128000},
        "gpt-4.5-orion": {"max_tokens": 8192, "context_window": 128000},
        
        # GPT-4o Series (Multimodal)
        "gpt-4o": {"max_tokens": 16384, "context_window": 128000},
        "gpt-4o-2024-11-20": {"max_tokens": 16384, "context_window": 128000},
        "gpt-4o-2024-08-06": {"max_tokens": 16384, "context_window": 128000},
        "gpt-4o-2024-05-13": {"max_tokens": 4096, "context_window": 128000},
        "gpt-4o-mini": {"max_tokens": 16384, "context_window": 128000},
        "gpt-4o-mini-2024-07-18": {"max_tokens": 16384, "context_window": 128000},
        
        # GPT-4 Turbo
        "gpt-4-turbo": {"max_tokens": 4096, "context_window": 128000},
        "gpt-4-turbo-2024-04-09": {"max_tokens": 4096, "context_window": 128000},
        "gpt-4-turbo-preview": {"max_tokens": 4096, "context_window": 128000},
        "gpt-4-0125-preview": {"max_tokens": 4096, "context_window": 128000},
        "gpt-4-1106-preview": {"max_tokens": 4096, "context_window": 128000},
        
        # GPT-4 Standard
        "gpt-4": {"max_tokens": 8192, "context_window": 8192},
        "gpt-4-0613": {"max_tokens": 8192, "context_window": 8192},
        
        # GPT-3.5 Turbo (Legacy)
        "gpt-3.5-turbo": {"max_tokens": 4096, "context_window": 16385},
        "gpt-3.5-turbo-16k": {"max_tokens": 16384, "context_window": 16385},
        "gpt-3.5-turbo-0125": {"max_tokens": 4096, "context_window": 16385},
        
        # Specialized Models
        "gpt-realtime-mini": {"max_tokens": 4096, "context_window": 16385},  # Voice model
        "gpt-image-1-mini": {"max_tokens": None, "context_window": None},     # Image generation
    }

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration with OpenAI-specific settings
        """
        # CRITICAL: Set attributes FIRST before anything that might fail
        self.session = None
        self.base_url = self.BASE_URL
        self.api_key = None
        
        # Extract config values safely
        if config:
            self.base_url = config.base_url or self.BASE_URL
            self.api_key = config.api_key
        
        # Now call parent init (which calls _validate_config)
        super().__init__(config)

    def _validate_config(self) -> None:
        """
        Validate OpenAI-specific configuration.

        Raises:
            ValueError: If configuration is invalid
            ModelNotFoundError: If model is not supported
        """
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        if self.config.model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model {self.config.model} not in known models. "
                f"Known models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Check token limits for known models
        if self.config.model in self.SUPPORTED_MODELS:
            model_info = self.SUPPORTED_MODELS[self.config.model]
            if model_info["max_tokens"] and self.config.max_tokens > model_info["max_tokens"]:
                logger.warning(
                    f"max_tokens {self.config.max_tokens} exceeds model limit "
                    f"{model_info['max_tokens']}, capping to model limit"
                )
                self.config.max_tokens = model_info["max_tokens"]

    async def _get_session(self):
        """Get or create aiohttp session."""
        if not aiohttp:
            raise LLMError("aiohttp is required for OpenAI provider", "openai", self.model)

        if not self.session or self.session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)

        return self.session

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                    **kwargs) -> LLMResponse:
        """Generate response using OpenAI API with GPT-5 support."""
        start_time = time.perf_counter()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False
        }
        
        # CRITICAL FIX: Handle GPT-5's different parameters
        model_lower = self.config.model.lower()
        max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)
        temperature_value = kwargs.get("temperature", self.config.temperature)
        
        if "gpt-5" in model_lower or "o1" in model_lower or "o3" in model_lower:
            # GPT-5+ uses max_completion_tokens and doesn't support custom temperature
            payload["max_completion_tokens"] = max_tokens_value
            # Don't set temperature for GPT-5 (only supports default 1.0)
            logger.debug(f"GPT-5 mode: max_completion_tokens={max_tokens_value}, no temperature")
        else:
            # GPT-4 and earlier use max_tokens and support custom temperature
            payload["max_tokens"] = max_tokens_value
            payload["temperature"] = temperature_value
            logger.debug(f"GPT-4 mode: max_tokens={max_tokens_value}, temperature={temperature_value}")

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]

        try:
            session = await self._get_session()

            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                response_data = await response.json()

                if response.status == 401:
                    raise AuthenticationError("Invalid OpenAI API key", "openai", self.model)
                elif response.status == 404:
                    raise ModelNotFoundError(f"Model {self.model} not found", "openai", self.model)
                elif response.status == 429:
                    raise RateLimitError("OpenAI rate limit exceeded", "openai", self.model)
                elif response.status != 200:
                    error_msg = response_data.get("error", {}).get("message", "Unknown error")
                    raise LLMError(f"OpenAI API error: {error_msg}", "openai", self.model)

                # Parse response
                choice = response_data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice["finish_reason"]

                usage = response_data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

                response_time_ms = (time.perf_counter() - start_time) * 1000

                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider="openai",
                    tokens_used=tokens_used,
                    response_time_ms=response_time_ms,
                    finish_reason=finish_reason,
                    metadata={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "request_id": response.headers.get("x-request-id", ""),
                        "model": response_data.get("model", self.config.model),
                        "created": response_data.get("created", 0)
                    }
                )

        except aiohttp.ClientError as e:
            raise LLMError(f"Network error: {e}", "openai", self.model)
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON response: {e}", "openai", self.model)
        except Exception as e:
            if isinstance(e, (LLMError, RateLimitError, AuthenticationError, ModelNotFoundError)):
                raise
            raise LLMError(f"Unexpected error: {e}", "openai", self.model)

    async def batch_generate(self, prompts: List[str],
                           system_prompt: Optional[str] = None,
                           **kwargs) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt for context
            **kwargs: Additional parameters

        Returns:
            List of LLMResponse objects
        """
        # OpenAI has stricter rate limits, so use lower concurrency
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests

        async def bounded_generate(prompt):
            async with semaphore:
                return await self.generate(prompt, system_prompt, **kwargs)

        tasks = [bounded_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough approximation. For accurate counting,
        use tiktoken library.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token for English
        # OpenAI's tiktoken would be more accurate
        return len(text) // 4

    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Send a minimal request to check connectivity
            response = await self.generate(
                "Test",
                max_tokens=1,
                temperature=0
            )
            return response.content is not None
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP session."""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            await self.session.close()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        model_info = self.SUPPORTED_MODELS.get(self.config.model, {})
        return {
            **self.get_provider_info(),
            "context_window": model_info.get("context_window", 0),
            "model_max_tokens": model_info.get("max_tokens", 0)
        }

    def __del__(self):
        """Cleanup when object is destroyed."""
        # DEFENSIVE: Check if session exists and is valid
        if hasattr(self, 'session') and self.session:
            try:
                if not self.session.closed:
                    # Cannot await in __del__, so we schedule it
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(self.session.close())
                    except RuntimeError:
                        # Event loop not running or closed
                        pass
            except Exception:
                # Silently ignore cleanup errors in destructor
                pass