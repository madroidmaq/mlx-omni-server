"""
Prompt Cache Management Module

This module provides functionality for managing and optimizing model prompt caching,
to improve performance in multi-turn conversations.
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple

from ...utils.logger import logger


@dataclass
class PromptCache:
    """
    Prompt cache class for storing and managing model prompt caches

    Attributes:
        tokens: Cached token sequence
        cache: Model's KV cache state, a list matching the number of model layers
        model_key: Model identifier to ensure cache matches the model
    """

    tokens: List[int] = field(default_factory=list)
    cache: List[Any] = field(default_factory=list)
    model_key: str = ""


def update_prompt_cache(
    prompt_cache: PromptCache,
    tokenized_prompt: List[int],
    model_key: str,
    cache_state: List[Any] = None,
) -> None:
    """
    Update prompt cache

    Args:
        prompt_cache: Prompt cache object
        tokenized_prompt: List of encoded prompt tokens
        model_key: Model identifier
        cache_state: Model's KV cache state, updates cache if provided
    """
    prompt_cache.tokens = tokenized_prompt.copy()
    prompt_cache.model_key = model_key

    # Update cache if new cache state is provided
    if cache_state is not None:
        prompt_cache.cache = cache_state
        logger.debug(f"Updated cache state with {len(cache_state)} layers")

    logger.debug(f"Updated cache with {len(tokenized_prompt)} tokens")


def process_prompt_cache(
    prompt: List[int], prompt_cache: PromptCache, model_key: str, model: Any
) -> Tuple[List[int], int]:
    """
    Process prompt cache using official logic

    Args:
        prompt: List of encoded prompt tokens
        prompt_cache: Prompt cache object
        model_key: Model identifier
        model: Model object used to create cache

    Returns:
        Tuple[List[int], int]: Tuple containing:
            1. List of prompt tokens to process (if cached, only returns uncached portion)
            2. Number of tokens retrieved from cache
    """
    from mlx_lm.models.cache import make_prompt_cache

    cache_len = len(prompt_cache.tokens)
    prompt_len = len(prompt)
    logger.debug(f"Prompt length: {prompt_len}, Cache length: {cache_len}")

    if cache_len > 0 and prompt_len > 0:
        prefix_len = min(10, cache_len, prompt_len)
        logger.debug(f"Cache tokens prefix: {prompt_cache.tokens[:prefix_len]}")
        logger.debug(f"Prompt tokens prefix: {prompt[:prefix_len]}")

    if (
        prompt_cache.model_key != model_key
        or cache_len >= prompt_len
        or prompt_cache.tokens != prompt[:cache_len]
    ):
        logger.debug("Resetting cache based on official logic")
        prompt_cache.model_key = model_key
        prompt_cache.cache = make_prompt_cache(model)
        prompt_cache.tokens = []

        prompt_cache.tokens.extend(prompt)
        return prompt, 0
    else:
        result_prompt = prompt[cache_len:]
        cached_tokens = cache_len
        logger.debug(f"Using cache. Cached tokens: {cached_tokens}")

        prompt_cache.tokens.extend(result_prompt)
        return result_prompt, cached_tokens
