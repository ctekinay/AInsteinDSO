"""
Ollama LLM provider implementation.

Ollama provider for local LLM deployment. Enables running models like
Mistral, LLaMA, CodeLlama locally without external API dependencies.
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


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local model deployment.

    Supports models like:
    - mistral
    - llama2
    - codellama
    - phi
    - mixtral
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration with Ollama-specific settings
        """
        # CRITICAL: Set attributes FIRST before anything that might fail
        self.session = None
        self.base_url = self.DEFAULT_BASE_URL
        
        # Extract config values safely
        if config:
            self.base_url = config.base_url or self.DEFAULT_BASE_URL
        
        # Now call parent init (which calls _validate_config)
        super().__init__(config)

    def _validate_config(self) -> None:
        """
        Validate Ollama-specific configuration.

        Note: Ollama doesn't require API keys for local deployment
        """
        # Ollama doesn't require API key for local deployment
        if not self.config.model:
            raise ValueError("Model name is required for Ollama")

        # Warn about common issues
        if self.base_url == self.DEFAULT_BASE_URL:
            logger.info("Using default Ollama URL (localhost:11434). "
                       "Ensure Ollama server is running locally.")

    async def _get_session(self):
        """Get or create aiohttp session."""
        if not aiohttp:
            raise LLMError("aiohttp is required for Ollama provider", "ollama", self.model)

        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

        return self.session

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      **kwargs) -> LLMResponse:
        """
        Generate response using Ollama API.

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt for context
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: If generation fails
            ModelNotFoundError: If model is not found
        """
        start_time = time.perf_counter()

        # Build request payload for Ollama
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        # Add optional parameters
        if "top_p" in kwargs:
            payload["options"]["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["options"]["top_k"] = kwargs["top_k"]
        if "repeat_penalty" in kwargs:
            payload["options"]["repeat_penalty"] = kwargs["repeat_penalty"]

        try:
            session = await self._get_session()

            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 404:
                    # Check if it's a model not found error or server not running
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("error", "Model not found")
                        if "not found" in error_msg.lower():
                            raise ModelNotFoundError(
                                f"Model {self.model} not found in Ollama",
                                "ollama",
                                self.model
                            )
                    except json.JSONDecodeError:
                        pass
                    raise LLMError("Ollama server not accessible. Is it running?", "ollama", self.model)

                elif response.status != 200:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("error", "Unknown error")
                    except json.JSONDecodeError:
                        error_msg = f"HTTP {response.status}"
                    raise LLMError(f"Ollama API error: {error_msg}", "ollama", self.model)

                response_data = await response.json()

                # Parse Ollama response
                content = response_data.get("response", "")
                finish_reason = "stop" if response_data.get("done", False) else "length"

                # Ollama provides some timing and token info
                eval_count = response_data.get("eval_count", 0)
                prompt_eval_count = response_data.get("prompt_eval_count", 0)
                total_tokens = eval_count + prompt_eval_count

                response_time_ms = (time.perf_counter() - start_time) * 1000

                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider="ollama",
                    tokens_used=total_tokens,
                    response_time_ms=response_time_ms,
                    finish_reason=finish_reason,
                    metadata={
                        "prompt_tokens": prompt_eval_count,
                        "completion_tokens": eval_count,
                        "eval_duration": response_data.get("eval_duration", 0),
                        "prompt_eval_duration": response_data.get("prompt_eval_duration", 0),
                        "total_duration": response_data.get("total_duration", 0),
                        "context": response_data.get("context", [])
                    }
                )

        except aiohttp.ClientError as e:
            raise LLMError(f"Network error connecting to Ollama: {e}", "ollama", self.model)
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON response from Ollama: {e}", "ollama", self.model)
        except Exception as e:
            if isinstance(e, (LLMError, ModelNotFoundError)):
                raise
            raise LLMError(f"Unexpected error with Ollama: {e}", "ollama", self.model)

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
        # Ollama can handle moderate concurrency for local deployment
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests for local server

        async def bounded_generate(prompt):
            async with semaphore:
                return await self.generate(prompt, system_prompt, **kwargs)

        tasks = [bounded_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4

    async def health_check(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            session = await self._get_session()

            # Check if Ollama server is running
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    # Check if our model is available
                    tags_data = await response.json()
                    models = [model["name"] for model in tags_data.get("models", [])]

                    # Check if exact model or base model exists
                    model_available = any(
                        self.config.model in model_name or model_name.startswith(self.config.model.split(':')[0])
                        for model_name in models
                    )

                    if not model_available:
                        logger.warning(
                            f"Model {self.config.model} not found in Ollama. "
                            f"Available models: {models}"
                        )
                        return False

                    return True
                else:
                    return False

        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.

        Returns:
            List of model dictionaries with name, size, etc.
        """
        try:
            session = await self._get_session()

            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    raise LLMError(
                        f"Failed to list models: HTTP {response.status}",
                        "ollama",
                        self.model
                    )

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Error listing Ollama models: {e}", "ollama", self.model)

    async def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull/download a model in Ollama.

        Args:
            model_name: Model to pull (defaults to current model)

        Returns:
            True if successful, False otherwise
        """
        target_model = model_name or self.config.model

        try:
            session = await self._get_session()

            payload = {"name": target_model}

            # Note: This is a streaming endpoint, but we'll wait for completion
            async with session.post(f"{self.base_url}/api/pull", json=payload) as response:
                if response.status == 200:
                    # Read the streaming response to completion
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("status") == "success":
                                    logger.info(f"Successfully pulled model {target_model}")
                                    return True
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"Failed to pull model {target_model}: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Error pulling model {target_model}: {e}")
            return False

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
        return {
            **self.get_provider_info(),
            "base_url": self.base_url,
            "local_deployment": True
        }

    def __del__(self):
        """Cleanup resources when object is destroyed."""
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