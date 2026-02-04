"""
Secrets configuration for accessing HuggingFace models.

This module provides secure access to API tokens without hardcoding them.
Tokens are loaded from environment variables or a local .env file.

Usage:
    from config.secrets import get_hf_token, get_sam3_processor

    # Get the token directly
    token = get_hf_token()

    # Or get a configured Sam3Processor
    processor = get_sam3_processor()
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False


def _load_env_file():
    """Load environment variables from .env file if it exists."""
    if _HAS_DOTENV:
        # Look for .env in project root
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return True
    return False


# Load .env on module import
_load_env_file()


def get_hf_token(required: bool = True) -> Optional[str]:
    """
    Get the HuggingFace access token.

    The token is read from the following sources (in order):
    1. HF_TOKEN environment variable
    2. HUGGINGFACE_TOKEN environment variable
    3. HUGGINGFACE_HUB_TOKEN environment variable

    Args:
        required: If True, raises an error when token is not found.
                  If False, returns None when token is not found.

    Returns:
        The HuggingFace access token, or None if not required and not found.

    Raises:
        EnvironmentError: If required=True and no token is found.
    """
    # Check multiple common environment variable names
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    if token is None and required:
        raise EnvironmentError(
            "HuggingFace access token not found. Please set one of the following:\n"
            "  1. Create a .env file in the project root with: HF_TOKEN=your_token_here\n"
            "  2. Set the HF_TOKEN environment variable\n"
            "\n"
            "You can get a token from: https://huggingface.co/settings/tokens"
        )

    return token


@lru_cache(maxsize=1)
def get_sam3_processor(model_name: str = "facebook/sam3-base"):
    """
    Get a configured Sam3Processor with authentication.

    The processor is cached after first load to avoid repeated initialization.

    Args:
        model_name: The HuggingFace model identifier for Sam3.

    Returns:
        A configured Sam3Processor instance.
    """
    from transformers import Sam3Processor

    token = get_hf_token(required=True)
    return Sam3Processor.from_pretrained(model_name, token=token)


def clear_processor_cache():
    """Clear the cached Sam3Processor (useful for testing)."""
    get_sam3_processor.cache_clear()
