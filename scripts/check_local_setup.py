#!/usr/bin/env python3
"""
Local Setup Checker for AInstein Alliander

Validates that your local LLM deployment is properly configured
for running EA Assistant without external API calls.

Usage:
    python scripts/check_local_setup.py

This script checks:
1. Environment configuration (.env file)
2. Local LLM endpoint connectivity
3. Required data files and directories
4. Python dependencies
5. Embedding model availability
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv


class Colors:
    """Terminal colors for output formatting."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_status(message: str, status: str, details: str = ""):
    """Print formatted status message."""
    if status == "✓":
        color = Colors.GREEN
    elif status == "✗":
        color = Colors.RED
    elif status == "⚠":
        color = Colors.YELLOW
    else:
        color = Colors.BLUE

    print(f"{color}{status}{Colors.END} {message}")
    if details:
        print(f"   {details}")


def check_env_file() -> Tuple[bool, Dict[str, str]]:
    """Check if .env file exists and load environment variables."""
    print_status("Checking environment configuration...", "🔍")

    env_path = Path(".env")
    if not env_path.exists():
        print_status(".env file not found", "✗",
                    "Copy .env.local to .env: cp .env.local .env")
        return False, {}

    # Load environment variables
    load_dotenv()

    required_vars = {
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
        "USE_OPENAI_EMBEDDINGS": os.getenv("USE_OPENAI_EMBEDDINGS", "false"),
        "ENABLE_API_RERANKING": os.getenv("ENABLE_API_RERANKING", "false"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    }

    # Provider-specific variables
    provider = required_vars["LLM_PROVIDER"]
    if provider == "ollama":
        required_vars.update({
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL")
        })
    elif provider == "openai":
        required_vars.update({
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL")
        })
    elif provider == "custom":
        required_vars.update({
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL")
        })

    # Check for missing variables
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        print_status("Missing required environment variables", "✗",
                    f"Missing: {', '.join(missing)}")
        return False, required_vars

    print_status("Environment configuration loaded", "✓")
    return True, required_vars


def check_local_only_config(env_vars: Dict[str, str]) -> bool:
    """Verify configuration is set for local-only deployment."""
    print_status("Verifying local-only configuration...", "🔍")

    checks = []

    # Check embeddings are local
    use_openai_embeddings = env_vars.get("USE_OPENAI_EMBEDDINGS", "false").lower()
    if use_openai_embeddings == "false":
        checks.append(("Local embeddings enabled", True))
    else:
        checks.append(("Local embeddings enabled", False,
                      "Set USE_OPENAI_EMBEDDINGS=false"))

    # Check API reranking is disabled
    enable_api_reranking = env_vars.get("ENABLE_API_RERANKING", "false").lower()
    if enable_api_reranking == "false":
        checks.append(("API reranking disabled", True))
    else:
        checks.append(("API reranking disabled", False,
                      "Set ENABLE_API_RERANKING=false"))

    # Check embedding model is local
    embedding_model = env_vars.get("EMBEDDING_MODEL", "")
    if "openai" not in embedding_model.lower():
        checks.append(("Local embedding model configured", True))
    else:
        checks.append(("Local embedding model configured", False,
                      "Use local model like 'all-MiniLM-L6-v2'"))

    all_passed = True
    for check in checks:
        if check[1]:
            print_status(check[0], "✓")
        else:
            print_status(check[0], "✗", check[2] if len(check) > 2 else "")
            all_passed = False

    return all_passed


def check_llm_endpoint(env_vars: Dict[str, str]) -> bool:
    """Test connectivity to local LLM endpoint."""
    print_status("Testing LLM endpoint connectivity...", "🔍")

    provider = env_vars.get("LLM_PROVIDER")

    if provider == "ollama":
        base_url = env_vars.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = env_vars.get("OLLAMA_MODEL")
        return test_ollama_endpoint(base_url, model)

    elif provider in ["openai", "custom"]:
        base_url = env_vars.get("OPENAI_BASE_URL", "http://localhost:1234")
        model = env_vars.get("OPENAI_MODEL")
        return test_openai_compatible_endpoint(base_url, model)

    else:
        print_status(f"Unknown provider: {provider}", "✗",
                    "Supported: ollama, openai, custom")
        return False


def test_ollama_endpoint(base_url: str, model: str) -> bool:
    """Test Ollama endpoint."""
    try:
        # Test health endpoint
        health_url = f"{base_url.rstrip('/')}/api/version"
        response = requests.get(health_url, timeout=5)

        if response.status_code != 200:
            print_status("Ollama server not responding", "✗",
                        f"GET {health_url} returned {response.status_code}")
            return False

        print_status("Ollama server responding", "✓")

        # Test model availability
        models_url = f"{base_url.rstrip('/')}/api/tags"
        response = requests.get(models_url, timeout=5)

        if response.status_code == 200:
            models_data = response.json()
            available_models = [m["name"] for m in models_data.get("models", [])]

            if model in available_models:
                print_status(f"Model '{model}' available", "✓")
                return True
            else:
                print_status(f"Model '{model}' not found", "✗",
                            f"Available: {', '.join(available_models)}")
                print_status("", "ℹ", f"Pull model: ollama pull {model}")
                return False
        else:
            print_status("Could not list models", "⚠",
                        "Server responding but model list unavailable")
            return True  # Assume OK if server is up

    except requests.RequestException as e:
        print_status("Cannot connect to Ollama server", "✗",
                    f"Error: {e}")
        print_status("", "ℹ", f"Ensure Ollama is running: ollama serve")
        return False


def test_openai_compatible_endpoint(base_url: str, model: str) -> bool:
    """Test OpenAI-compatible endpoint."""
    try:
        # Test models endpoint
        models_url = f"{base_url.rstrip('/')}/v1/models"
        response = requests.get(models_url, timeout=5)

        if response.status_code == 200:
            print_status("LLM server responding", "✓")

            models_data = response.json()
            available_models = [m["id"] for m in models_data.get("data", [])]

            if model in available_models:
                print_status(f"Model '{model}' available", "✓")
            else:
                print_status(f"Model '{model}' not listed", "⚠",
                            f"Available: {', '.join(available_models)}")
                print_status("", "ℹ", "Check your model name configuration")

            return True

        elif response.status_code == 404:
            # Try simple health check
            health_url = f"{base_url.rstrip('/')}/health"
            health_response = requests.get(health_url, timeout=5)

            if health_response.status_code == 200:
                print_status("LLM server responding", "✓")
                print_status("Model listing not available", "⚠",
                            "Server up but no /v1/models endpoint")
                return True

        print_status("LLM server not responding properly", "✗",
                    f"GET {models_url} returned {response.status_code}")
        return False

    except requests.RequestException as e:
        print_status("Cannot connect to LLM server", "✗",
                    f"Error: {e}")
        print_status("", "ℹ", f"Ensure your LLM server is running at {base_url}")
        return False


def check_data_files() -> bool:
    """Check required data files and directories exist."""
    print_status("Checking data files and directories...", "🔍")

    required_paths = [
        ("data/", "directory", "Data directory"),
        ("data/energy_knowledge_graph.ttl", "file", "Knowledge graph"),
        ("config/vocabularies.json", "file", "Vocabularies config"),
    ]

    optional_paths = [
        ("data/models/", "directory", "ArchiMate models"),
        ("data/docs/", "directory", "PDF documents"),
        ("data/embeddings/", "directory", "Embeddings cache"),
    ]

    all_required = True

    for path_str, path_type, description in required_paths:
        path = Path(path_str)

        if path_type == "directory" and path.is_dir():
            print_status(f"{description} found", "✓")
        elif path_type == "file" and path.is_file():
            print_status(f"{description} found", "✓")
        else:
            print_status(f"{description} missing", "✗", f"Expected: {path_str}")
            all_required = False

    for path_str, path_type, description in optional_paths:
        path = Path(path_str)

        if path_type == "directory" and path.is_dir():
            print_status(f"{description} found", "✓")
        elif path_type == "file" and path.is_file():
            print_status(f"{description} found", "✓")
        else:
            print_status(f"{description} not found", "⚠",
                        f"Optional: {path_str}")

    return all_required


def check_python_dependencies() -> bool:
    """Check Python dependencies are installed."""
    print_status("Checking Python dependencies...", "🔍")

    required_packages = [
        "fastapi", "uvicorn", "rdflib", "sentence_transformers",
        "torch", "transformers", "numpy", "requests", "python-dotenv"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print_status("Missing Python packages", "✗",
                    f"Install: pip install {' '.join(missing)}")
        return False

    print_status("All required packages installed", "✓")
    return True


def test_embedding_model(model_name: str) -> bool:
    """Test local embedding model availability."""
    print_status("Testing embedding model...", "🔍")

    try:
        from sentence_transformers import SentenceTransformer

        print_status(f"Loading model '{model_name}'...", "ℹ")
        model = SentenceTransformer(model_name)

        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)

        if len(embedding) > 0:
            print_status(f"Embedding model working", "✓",
                        f"Generated {len(embedding)}-dimensional vector")
            return True
        else:
            print_status("Embedding generation failed", "✗")
            return False

    except Exception as e:
        print_status("Embedding model error", "✗", f"Error: {e}")
        print_status("", "ℹ", f"Model will be downloaded on first use")
        return True  # Don't fail setup for this


def run_integration_test(env_vars: Dict[str, str]) -> bool:
    """Run basic integration test."""
    print_status("Running integration test...", "🔍")

    try:
        # Try to import and initialize key components
        from agents.embedding_agent import EmbeddingAgent

        print_status("Initializing EA Assistant...", "ℹ")
        agent = EmbeddingAgent()

        # Simple test query
        test_query = "What is a business capability?"
        print_status(f"Testing query: '{test_query}'", "ℹ")

        response = agent.process_query(test_query)

        if response and hasattr(response, 'answer') and response.answer:
            print_status("Integration test passed", "✓",
                        "EA Assistant responding normally")
            return True
        else:
            print_status("Integration test failed", "✗",
                        "EA Assistant not responding properly")
            return False

    except Exception as e:
        print_status("Integration test error", "✗", f"Error: {e}")
        return False


def main():
    """Main setup checker."""
    print(f"{Colors.BOLD}🔧 AInstein Local Setup Checker{Colors.END}")
    print("=" * 50)

    all_checks = []

    # Check environment
    env_ok, env_vars = check_env_file()
    all_checks.append(env_ok)
    print()

    if env_ok:
        # Check local-only configuration
        local_config_ok = check_local_only_config(env_vars)
        all_checks.append(local_config_ok)
        print()

        # Check LLM endpoint
        llm_ok = check_llm_endpoint(env_vars)
        all_checks.append(llm_ok)
        print()

    # Check data files
    data_ok = check_data_files()
    all_checks.append(data_ok)
    print()

    # Check Python dependencies
    deps_ok = check_python_dependencies()
    all_checks.append(deps_ok)
    print()

    if env_ok:
        # Test embedding model
        embedding_model = env_vars.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embed_ok = test_embedding_model(embedding_model)
        all_checks.append(embed_ok)
        print()

        # Integration test (only if basics work)
        if all(all_checks):
            integration_ok = run_integration_test(env_vars)
            all_checks.append(integration_ok)
            print()

    # Summary
    print("=" * 50)
    passed = sum(all_checks)
    total = len(all_checks)

    if passed == total:
        print_status(f"Setup validation passed ({passed}/{total})", "✅")
        print(f"\n{Colors.GREEN}🎉 Your local setup is ready!{Colors.END}")
        print(f"\nNext steps:")
        print(f"  • Start EA Assistant: python run_web_demo.py")
        print(f"  • Open browser: http://localhost:8000")
        print(f"  • Test CLI: python test_conversation.py")
    else:
        print_status(f"Setup validation failed ({passed}/{total})", "❌")
        print(f"\n{Colors.RED}⚠️  Please fix the issues above{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()