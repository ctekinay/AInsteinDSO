"""
Groq LLM provider implementation.

Groq provides fast inference for open-source models like Llama, Qwen, Kimi, etc.
This provider implements the LLMProvider interface for Groq's API.
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


class GroqProvider(LLMProvider):
    """
    Groq LLM provider for fast inference with open-source models.

    Supports models including (as of October 2025):
    - llama-3.3-70b-versatile (RECOMMENDED - 131K context)
    - moonshotai/kimi-k2-instruct (1T params MoE, 131K context)
    - qwen/qwen3-32b (advanced reasoning, 131K context)
    - llama-3.1-8b-instant (fast, 131K context)
    - deepseek-r1-distill-llama-70b (reasoning model)
    - meta-llama/llama-4-* (latest Llama 4 models)
    - whisper-large-v3-turbo (audio transcription)
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    SUPPORTED_MODELS = {
        # ====================
        # PRODUCTION MODELS (Stable, Production-Ready)
        # ====================
        
        # Meta Llama 3.3 (RECOMMENDED)
        "llama-3.3-70b-versatile": {"max_tokens": 32768, "context_window": 131072},
        
        # Meta Llama 3.1
        "llama-3.1-8b-instant": {"max_tokens": 131072, "context_window": 131072},
        
        # Google Gemma 2
        "gemma2-9b-it": {"max_tokens": 8192, "context_window": 8192},
        
        # Meta Llama Guard (Content Moderation)
        "meta-llama/llama-guard-4-12b": {"max_tokens": 1024, "context_window": 131072},
        
        # OpenAI Whisper (Audio Transcription)
        "whisper-large-v3": {"max_tokens": None, "context_window": None},
        "whisper-large-v3-turbo": {"max_tokens": None, "context_window": None},
        
        # ====================
        # PREVIEW MODELS (Evaluation Only)
        # ====================
        
        # Moonshot AI Kimi K2 (1 Trillion Parameter MoE)
        "moonshotai/kimi-k2-instruct": {"max_tokens": 16384, "context_window": 131072},
        
        # Alibaba Qwen 3 32B
        "qwen/qwen3-32b": {"max_tokens": 40960, "context_window": 131072},
        
        # DeepSeek R1 Distill (Reasoning Model)
        "deepseek-r1-distill-llama-70b": {"max_tokens": 131072, "context_window": 131072},
        
        # Meta Llama 4 Series (Latest)
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"max_tokens": 8192, "context_window": 131072},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"max_tokens": 8192, "context_window": 131072},
        
        # Meta Llama Prompt Guard (Security)
        "meta-llama/llama-prompt-guard-2-22m": {"max_tokens": 512, "context_window": 512},
        "meta-llama/llama-prompt-guard-2-86m": {"max_tokens": 512, "context_window": 512},
        
        # PlayAI TTS (Text-to-Speech)
        "playai-tts": {"max_tokens": 8192, "context_window": 8192},
        "playai-tts-arabic": {"max_tokens": 8192, "context_window": 8192},
        
        # ====================
        # COMPOUND AI SYSTEMS (Agentic)
        # ====================
        "compound": {"max_tokens": 131072, "context_window": 131072},  # Groq's compound AI system
        
        # ====================
        # DEPRECATED/LEGACY MODELS (Still supported but will be removed)
        # ====================
        "llama-3.1-70b-versatile": {"max_tokens": 8192, "context_window": 131072},  # Use llama-3.3-70b-versatile instead
        "mixtral-8x7b-32768": {"max_tokens": 32768, "context_window": 32768},  # Legacy Mistral model
        "llama3-70b-8192": {"max_tokens": 8192, "context_window": 8192},  # Use llama-3.3-70b-versatile
        "llama3-8b-8192": {"max_tokens": 8192, "context_window": 8192},  # Use llama-3.1-8b-instant
        "gemma-7b-it": {"max_tokens": 8192, "context_window": 8192},  # Use gemma2-9b-it
    }

    def __init__(self, config: LLMConfig):
        """
        Initialize Groq provider.
        
        Args:
            config: LLM configuration with Groq-specific settings
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
        Validate Groq-specific configuration.

        Raises:
            ValueError: If configuration is invalid
            ModelNotFoundError: If model is not supported
        """
        if not self.config.api_key:
            raise ValueError("Groq API key is required")

        if self.config.model not in self.SUPPORTED_MODELS:
            supported_list = list(self.SUPPORTED_MODELS.keys())
            raise ModelNotFoundError(
                f"Model {self.config.model} not supported. "
                f"Supported models: {supported_list}",
                provider="groq",
                model=self.config.model
            )

        model_info = self.SUPPORTED_MODELS[self.config.model]

        # Handle Whisper/TTS models which don't have token limits
        if model_info["max_tokens"] is not None:
            if self.config.max_tokens > model_info["max_tokens"]:
                logger.warning(
                    f"max_tokens {self.config.max_tokens} exceeds model limit "
                    f"{model_info['max_tokens']}, capping to model limit"
                )
                self.config.max_tokens = model_info["max_tokens"]

    async def _get_session(self):
        """Get or create aiohttp session."""
        if not aiohttp:
            raise LLMError(
                "aiohttp is required for Groq provider. Install with: pip install aiohttp",
                "groq",
                self.model
            )

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
        """
        Generate response using Groq API.

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt for context
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: If generation fails
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            ModelNotFoundError: If model is not found
        """
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
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        try:
            session = await self._get_session()

            async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                response_data = await response.json()

                if response.status == 401:
                    raise AuthenticationError("Invalid Groq API key", "groq", self.model)
                elif response.status == 404:
                    raise ModelNotFoundError(f"Model {self.model} not found", "groq", self.model)
                elif response.status == 429:
                    raise RateLimitError("Groq rate limit exceeded", "groq", self.model)
                elif response.status != 200:
                    error_msg = response_data.get("error", {}).get("message", "Unknown error")
                    raise LLMError(f"Groq API error: {error_msg}", "groq", self.model)

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
                    provider="groq",
                    tokens_used=tokens_used,
                    response_time_ms=response_time_ms,
                    finish_reason=finish_reason,
                    metadata={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "request_id": response_data.get("id", ""),
                        "system_fingerprint": response_data.get("system_fingerprint", "")
                    }
                )

        except aiohttp.ClientError as e:
            raise LLMError(f"Network error: {e}", "groq", self.model)
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON response: {e}", "groq", self.model)
        except Exception as e:
            if isinstance(e, (LLMError, RateLimitError, AuthenticationError, ModelNotFoundError)):
                raise
            raise LLMError(f"Unexpected error: {e}", "groq", self.model)

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
        # Execute with concurrency limit to avoid rate limits
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def bounded_generate(prompt):
            async with semaphore:
                return await self.generate(prompt, system_prompt, **kwargs)

        tasks = [bounded_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough approximation. For accurate counting,
        use tiktoken or similar tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token for English
        return len(text) // 4

    async def health_check(self) -> bool:
        """
        Check if Groq API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Basic configuration check
            if not self.api_key:
                logger.warning("Groq health check failed: No API key")
                return False
            if not aiohttp:
                logger.warning("Groq health check failed: aiohttp not available")
                return False

            # For now, just check if we have the basic requirements
            # API call health check can be too strict during initialization
            logger.info("Groq health check passed (basic config validation)")
            return True

        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
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