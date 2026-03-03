"""
Prompt Cache Pool Module

This module provides functionality for managing PromptCache instances, avoiding concurrency issues.
"""

import time
from dataclasses import dataclass, field
from typing import List
import copy

from ...utils.logger import logger
from .prompt_cache import PromptCache, common_prefix_len


@dataclass
class PoolEntry:
    """A single entry in the prompt cache pool."""

    cache: PromptCache
    last_used: float = field(default_factory=time.time)

class PromptCachePool:
    """
    Prompt cache pool class for managing PromptCache instances using LRU policy. 
    """

    def __init__(self, max_size: int = 8, ttl_seconds: float = 300.0):
        self._entries: List[PoolEntry] = []
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def get_cache(self, prompt_tokens: List[int], model_key: str) -> PromptCache:
        """Check out a deepcopy of the best matching PromptCache 

        Finds the best matching available cache by common prefix length.
        If no suitable cache exists, creates a new one (evicting the oldest
        entry if at capacity). 

        Args:
            prompt_tokens: The tokenized prompt for this request.
            model_key: The model identifier to match against cached model_key.

        Returns:
            A PromptCache instance that the caller has exclusive access to.
        """
        self._evict_expired()

        best_entry = None
        best_prefix_len = -1

        for entry in self._entries:
            if entry.cache.model_key != model_key:
                continue
            prefix_len = common_prefix_len(entry.cache.tokens, prompt_tokens)
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_entry = entry

        # Retrieves best matched entry
        if best_entry is not None:
            best_entry.last_used = time.time()
            logger.debug(
                f"Pool checkout: reusing cache with {best_prefix_len} "
                f"prefix tokens (pool size: {len(self._entries)})"
            )
            return copy.deepcopy(best_entry.cache)

        # No suitable match — create new entry
        if len(self._entries) >= self._max_size:
            # First evict if at capacity
            self._evict_oldest()

        new_entry = PoolEntry(
            cache=PromptCache(),
            last_used=time.time(),
        )
        self._entries.append(new_entry)
        logger.debug(
            f"Pool checkout: new cache created "
            f"(pool size: {len(self._entries)})"
        )
        return new_entry.cache

    def put_cache(self, cache: PromptCache) -> None:
        """Registers a PromptCache after use.

        Args:
            cache: The PromptCache instance to register.
        """
        self._entries.append(
            PoolEntry(
                cache=cache,
                last_used=time.time(),
            )
        )

        if len(self._entries) > self._max_size:
            self._evict_oldest()

    def _evict_expired(self) -> None:
        """Remove entries that have exceeded TTL. """
        if self._ttl_seconds <= 0:
            return
        now = time.time()
        before = len(self._entries)
        self._entries = [
            e
            for e in self._entries
            if now - e.last_used <= self._ttl_seconds
        ]
        evicted = before - len(self._entries)
        if evicted > 0:
            logger.debug(f"Pool evicted {evicted} expired cache(s)")

    def _evict_oldest(self) -> None:
        """Remove the oldest entry. """
        oldest_idx = None
        oldest_time = float("inf")
        for i, entry in enumerate(self._entries):
            if entry.last_used < oldest_time:
                oldest_time = entry.last_used
                oldest_idx = i
        if oldest_idx is not None:
            removed = self._entries.pop(oldest_idx)
            logger.debug(
                f"Pool evicted cache (tokens: {len(removed.cache.tokens)})"
            )

    def get_pool_info(self) -> dict:
        """Get pool statistics for debugging/monitoring."""
        return {
            "pool_size": len(self._entries),
            "max_size": self._max_size,
            "utilization": len(self._entries) / self._max_size,
            "ttl_seconds": self._ttl_seconds,
        }
