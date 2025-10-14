"""
LLM Factory for provider-agnostic LLM integration.

This factory handles provider selection, fallback logic, and configuration
management for different LLM providers (Groq, OpenAI, Ollama).

UPDATED: Support for dual Groq API keys (GROQ_API_KEY_1 and GROQ_API_KEY_2)
for LLM Council implementation.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Type, Any
from enum import Enum

from .base import LLMProvider, LLMConfig, LLMError
from .groq_provider import GroqProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Enumeration of supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMFactory:
    """
    Factory for creating LLM providers with fallback support.

    Handles provider selection, configuration loading, and automatic
    fallback when primary providers are unavailable.
    
    UPDATED: Supports dual Groq API keys for LLM Council pattern.
    """

    PROVIDER_CLASSES: Dict[str, Type[LLMProvider]] = {
        ProviderType.GROQ.value: GroqProvider,
        ProviderType.OPENAI.value: OpenAIProvider,
        ProviderType.OLLAMA.value: OllamaProvider,
    }

    DEFAULT_CONFIGS = {
        ProviderType.GROQ.value: {
            "model": "llama-3.3-70b-versatile",  # UPDATED - Latest Groq model
            "max_tokens": 1024,
            "temperature": 0.3,
            "timeout": 30
        },
        ProviderType.OPENAI.value: {
            "model": "gpt-5",  # UPDATED - Latest OpenAI model
            "max_tokens": 1024,
            "temperature": 0.3,
            "timeout": 30
        },
        ProviderType.OLLAMA.value: {
            "model": "mistral",
            "max_tokens": 1024,
            "temperature": 0.3,
            "timeout": 60  # Longer timeout for local processing
        }
    }

    def __init__(self, env_file: Optional[str] = None, use_secondary_key: bool = False):
        """
        Initialize LLM factory.

        Args:
            env_file: Path to environment file (defaults to .env)
            use_secondary_key: If True, use GROQ_API_KEY_2 for Groq provider
        """
        self.env_file = env_file or ".env"
        self.use_secondary_key = use_secondary_key
        self.configs: Dict[str, LLMConfig] = {}
        self.providers: Dict[str, LLMProvider] = {}
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load configurations from environment variables and defaults."""

        # Load environment variables from .env file if it exists
        if os.path.exists(self.env_file):
            self._load_env_file()

        # Create configurations for each provider
        for provider_type in ProviderType:
            provider_name = provider_type.value
            config = self._create_provider_config(provider_name)
            if config:
                self.configs[provider_name] = config
                logger.info(f"Configured {provider_name} provider with model {config.model}")

    def _load_env_file(self) -> None:
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        except ImportError:
            logger.warning("python-dotenv not available, using manual parsing")
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")

    def _create_provider_config(self, provider: str) -> Optional[LLMConfig]:
        """
        Create configuration for a specific provider.
        
        FIXED: Correct parameter order for LLMConfig (api_key, model first).

        Args:
            provider: Provider name (groq, openai, ollama)

        Returns:
            LLMConfig if properly configured, None otherwise
        """
        provider_upper = provider.upper()

        # Get configuration from environment or defaults
        defaults = self.DEFAULT_CONFIGS.get(provider, {})

        # Special handling for Groq provider with dual API keys
        if provider == "groq":
            # Determine which API key to use
            if self.use_secondary_key:
                api_key = os.getenv("GROQ_API_KEY_2")
                model = os.getenv("GROQ_MODEL_VALIDATOR", defaults.get("model"))
                logger.info("Using secondary Groq API key (GROQ_API_KEY_2) for validation")
            else:
                # Try GROQ_API_KEY_1 first, fall back to GROQ_API_KEY for backward compatibility
                api_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
                model = os.getenv("GROQ_MODEL_PRIMARY") or os.getenv("GROQ_MODEL", defaults.get("model"))
                if os.getenv("GROQ_API_KEY_1"):
                    logger.info("Using primary Groq API key (GROQ_API_KEY_1)")
                else:
                    logger.info("Using legacy Groq API key (GROQ_API_KEY)")
            
            # FIXED: Correct parameter order
            config_data = {
                "api_key": api_key,    # FIRST - required
                "model": model,        # SECOND - required
                "provider": provider,  # THIRD - optional
                "base_url": os.getenv(f"{provider_upper}_BASE_URL"),
                "max_tokens": int(os.getenv(f"{provider_upper}_MAX_TOKENS", defaults.get("max_tokens", 1024))),
                "temperature": float(os.getenv(f"{provider_upper}_TEMPERATURE", defaults.get("temperature", 0.3))),
                "timeout": int(os.getenv(f"{provider_upper}_TIMEOUT", defaults.get("timeout", 30))),
                "retries": int(os.getenv(f"{provider_upper}_RETRIES", "3"))
            }
        else:
            # Standard configuration for other providers
            # FIXED: Correct parameter order
            config_data = {
                "api_key": os.getenv(f"{provider_upper}_API_KEY"),    # FIRST - required
                "model": os.getenv(f"{provider_upper}_MODEL", defaults.get("model")),  # SECOND - required
                "provider": provider,  # THIRD - optional
                "base_url": os.getenv(f"{provider_upper}_BASE_URL"),
                "max_tokens": int(os.getenv(f"{provider_upper}_MAX_TOKENS", defaults.get("max_tokens", 1024))),
                "temperature": float(os.getenv(f"{provider_upper}_TEMPERATURE", defaults.get("temperature", 0.3))),
                "timeout": int(os.getenv(f"{provider_upper}_TIMEOUT", defaults.get("timeout", 30))),
                "retries": int(os.getenv(f"{provider_upper}_RETRIES", "3"))
            }

        # Check if required fields are present
        if provider in ["groq", "openai"] and not config_data["api_key"]:
            logger.warning(f"No API key found for {provider} provider")
            return None

        if not config_data["model"]:
            logger.warning(f"No model specified for {provider} provider")
            return None

        # Filter out None values before creating LLMConfig
        return LLMConfig(**{k: v for k, v in config_data.items() if v is not None})

    async def create_provider(self, provider: str, config: Optional[LLMConfig] = None) -> LLMProvider:
        """
        Create a specific LLM provider.

        Args:
            provider: Provider name (groq, openai, ollama)
            config: Optional custom configuration

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider is not supported
            LLMError: If provider cannot be created
        """
        if provider not in self.PROVIDER_CLASSES:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported: {list(self.PROVIDER_CLASSES.keys())}")

        # Use provided config or stored config
        provider_config = config or self.configs.get(provider)
        if not provider_config:
            raise LLMError(f"No configuration found for provider {provider}", provider, "unknown")

        try:
            provider_class = self.PROVIDER_CLASSES[provider]
            provider_instance = provider_class(provider_config)

            # Test provider health
            is_healthy = await provider_instance.health_check()
            if not is_healthy:
                raise LLMError(f"Provider {provider} failed health check", provider,
                             provider_config.model)

            self.providers[provider] = provider_instance
            logger.info(f"Successfully created {provider} provider")
            return provider_instance

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Failed to create {provider} provider: {e}", provider,
                         provider_config.model if provider_config else "unknown")

    async def create_dual_groq_providers(self) -> tuple[Optional[LLMProvider], Optional[LLMProvider]]:
        """
        Create both primary and validator Groq providers for LLM Council.

        Returns:
            Tuple of (primary_provider, validator_provider)
        """
        primary_provider = None
        validator_provider = None

        try:
            # Create primary provider using GROQ_API_KEY_1
            factory_primary = LLMFactory(self.env_file, use_secondary_key=False)
            primary_provider = await factory_primary.create_provider("groq")
            logger.info("Created primary Groq provider")
        except Exception as e:
            logger.error(f"Failed to create primary Groq provider: {e}")

        try:
            # Create validator provider using GROQ_API_KEY_2
            factory_validator = LLMFactory(self.env_file, use_secondary_key=True)
            validator_provider = await factory_validator.create_provider("groq")
            logger.info("Created validator Groq provider")
        except Exception as e:
            logger.error(f"Failed to create validator Groq provider: {e}")

        return primary_provider, validator_provider

    async def create_with_fallback(self,
                                 primary: str = "groq",
                                 fallbacks: Optional[List[str]] = None) -> Optional[LLMProvider]:
        """
        Create LLM provider with automatic fallback.

        Args:
            primary: Primary provider to try first
            fallbacks: List of fallback providers (defaults to ['openai', 'ollama'])

        Returns:
            Working LLMProvider instance or None if all fail

        Note:
            Returns None instead of raising when no providers available,
            allowing graceful fallback to template responses.
        """
        if fallbacks is None:
            fallbacks = ["openai", "ollama"]

        providers_to_try = [primary] + [p for p in fallbacks if p != primary]

        for provider_name in providers_to_try:
            try:
                logger.info(f"Attempting to create {provider_name} provider...")
                provider = await self.create_provider(provider_name)
                logger.info(f"Successfully created {provider_name} provider")
                return provider

            except Exception as e:
                logger.warning(f"Failed to create {provider_name} provider: {e}")
                continue

        # Return None for graceful fallback instead of raising
        logger.info("All LLM providers failed, will use template fallback")
        return None

    def get_available_providers(self) -> List[str]:
        """
        Get list of configured providers.

        Returns:
            List of provider names that are configured
        """
        return list(self.configs.keys())

    def get_provider_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific provider.

        Args:
            provider: Provider name

        Returns:
            Provider information dictionary or None
        """
        config = self.configs.get(provider)
        if not config:
            return None

        info = {
            "provider": provider,
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "configured": True
        }

        # Add provider-specific info
        if provider == "groq":
            info["api_required"] = True
            info["speed"] = "fast"
            info["dual_keys_configured"] = bool(os.getenv("GROQ_API_KEY_1") and os.getenv("GROQ_API_KEY_2"))
        elif provider == "openai":
            info["api_required"] = True
            info["quality"] = "high"
        elif provider == "ollama":
            info["api_required"] = False
            info["deployment"] = "local"
            info["base_url"] = config.base_url

        return info

    async def close_all(self):
        """Close all created providers."""
        for provider in self.providers.values():
            try:
                await provider.close()
            except Exception as e:
                logger.warning(f"Error closing provider: {e}")

        self.providers.clear()

    def create_config_template(self, output_path: str = ".env.template") -> None:
        """
        Create a template .env file with all configuration options.

        UPDATED: Includes OpenAI GPT-5 and Groq Llama-3.3 configuration.

        Args:
            output_path: Path to write template file
        """
        template_content = """# LLM Provider Configuration

    # Primary LLM provider (groq, openai, ollama)
    LLM_PROVIDER=openai

    # OpenAI Configuration (Primary - GPT-5)
    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_MODEL=gpt-5
    OPENAI_MAX_TOKENS=4096
    OPENAI_TEMPERATURE=0.3
    OPENAI_TIMEOUT=60
    OPENAI_RETRIES=3

    # Groq Configuration (Validator - Llama 3.3)
    GROQ_API_KEY=your_groq_api_key_here
    GROQ_MODEL=llama-3.3-70b-versatile
    GROQ_MAX_TOKENS=4096
    GROQ_TEMPERATURE=0.2
    GROQ_TIMEOUT=30
    GROQ_RETRIES=3

    # Legacy Groq dual-key support (optional)
    GROQ_API_KEY_1=your_groq_api_key_1_here
    GROQ_API_KEY_2=your_groq_api_key_2_here
    GROQ_MODEL_PRIMARY=llama-3.3-70b-versatile
    GROQ_MODEL_VALIDATOR=qwen/qwen3-32b

    # Ollama Configuration (Local deployment fallback)
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=mistral
    OLLAMA_MAX_TOKENS=1024
    OLLAMA_TEMPERATURE=0.3
    OLLAMA_TIMEOUT=60
    OLLAMA_RETRIES=3

    # Performance Settings
    LLM_FALLBACK_ENABLED=true
    LLM_HEALTH_CHECK_TIMEOUT=10
    """

        with open(output_path, 'w') as f:
            f.write(template_content)

        logger.info(f"Created configuration template at {output_path}")


# Convenience function for easy provider creation
async def create_llm_provider(provider: Optional[str] = None,
                            config_file: Optional[str] = None,
                            use_secondary_key: bool = False) -> Optional[LLMProvider]:
    """
    Create LLM provider with automatic configuration and fallback.

    UPDATED: Added use_secondary_key parameter for dual Groq support.

    Args:
        provider: Specific provider to use (defaults to env LLM_PROVIDER or 'groq')
        config_file: Path to .env file
        use_secondary_key: If True and provider is groq, use GROQ_API_KEY_2

    Returns:
        Working LLMProvider instance or None if all providers fail
    """
    factory = LLMFactory(config_file, use_secondary_key)

    # Determine primary provider - default to groq if none specified
    primary_provider = provider or os.getenv("LLM_PROVIDER", "groq")

    # Create with fallback
    return await factory.create_with_fallback(primary_provider)