"""
Prompt Cache Pool - Thread-safe pool of PromptCache instances.

Provides exclusive checkout/checkin semantics so that concurrent requests
sharing the same ChatGenerator can each get their own PromptCache without
data races, while still benefiting from prefix-matching reuse across
multi-turn conversations.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List

from ...utils.logger import logger
from .prompt_cache import PromptCache, common_prefix_len


@dataclass
class PoolEntry:
    """A single entry in the prompt cache pool."""

    cache: PromptCache
    in_use: bool = False
    last_used: float = field(default_factory=time.time)

@dataclass
class CacheBlock:
    block_id: int
    block_hash: str
    tokens: tuple[int, ...]  # The tokens in this block
    kv_cache: any  # The actual mlx KV cache data
    ref_cnt: int = 0
    parent_hash: Optional[str] = None

class PromptCachePool:
    """Thread-safe pool of PromptCache instances with exclusive checkout.

    Maintains a pool of PromptCache instances. On checkout, finds the best
    prefix-matching idle cache and marks it in_use. On checkin, marks it
    available again. TTL eviction removes idle caches that haven't been
    accessed within ttl_seconds.

    Args:
        max_size: Maximum number of PromptCache instances in the pool.
        ttl_seconds: Time-to-live for idle cache entries.
    """

    def __init__(self, max_size: int = 4, ttl_seconds: float = 300.0):
        self._entries: List[PoolEntry] = []
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def checkout(self, prompt_tokens: List[int], model_key: str) -> PromptCache:
        """Check out a PromptCache with exclusive access.

        Finds the best matching available cache by common prefix length.
        If no suitable cache exists, creates a new one (evicting the oldest
        idle entry if at capacity). If the pool is full and all entries are
        in use, returns a fresh unmanaged PromptCache that won't block.

        Args:
            prompt_tokens: The tokenized prompt for this request.
            model_key: The model identifier to match against cached model_key.

        Returns:
            A PromptCache instance that the caller has exclusive access to.
        """
        with self._lock:
            self._evict_expired()

            best_entry = None
            best_prefix_len = -1

            for entry in self._entries:
                if entry.in_use:
                    continue
                if entry.cache.model_key != model_key:
                    continue
                prefix_len = common_prefix_len(entry.cache.tokens, prompt_tokens)
                if prefix_len > best_prefix_len:
                    best_prefix_len = prefix_len
                    best_entry = entry

            if best_entry is not None:
                best_entry.in_use = True
                best_entry.last_used = time.time()
                logger.debug(
                    f"Pool checkout: reusing cache with {best_prefix_len} "
                    f"prefix tokens (pool size: {len(self._entries)})"
                )
                return best_entry.cache

            # No suitable match — create new entry
            if len(self._entries) >= self._max_size:
                self._evict_oldest_idle()

            if len(self._entries) >= self._max_size:
                # All entries are in use and pool is full.
                # Return a temporary unmanaged cache so we never block.
                logger.warning(
                    f"Pool at capacity ({self._max_size}) with all entries in use. "
                    f"Creating temporary unmanaged cache."
                )
                return PromptCache()

            new_entry = PoolEntry(
                cache=PromptCache(),
                in_use=True,
                last_used=time.time(),
            )
            self._entries.append(new_entry)
            logger.debug(
                f"Pool checkout: new cache created "
                f"(pool size: {len(self._entries)})"
            )
            return new_entry.cache

    def checkin(self, cache: PromptCache) -> None:
        """Return a PromptCache to the pool after use.

        Args:
            cache: The PromptCache instance to return.
        """
        with self._lock:
            for entry in self._entries:
                if entry.cache is cache:
                    entry.in_use = False
                    entry.last_used = time.time()
                    logger.debug(
                        f"Pool checkin: cache returned "
                        f"(tokens: {len(cache.tokens)}, "
                        f"pool size: {len(self._entries)})"
                    )
                    return

            # Cache was not found in pool (temporary/unmanaged).
            # Adopt it if there's room, otherwise discard.
            if len(self._entries) < self._max_size:
                self._entries.append(
                    PoolEntry(
                        cache=cache,
                        in_use=False,
                        last_used=time.time(),
                    )
                )
                logger.debug(
                    f"Pool checkin: adopted temporary cache "
                    f"(pool size: {len(self._entries)})"
                )
            else:
                logger.debug(
                    "Pool checkin: discarding temporary cache (pool full)"
                )

    def _evict_expired(self) -> None:
        """Remove idle entries that have exceeded TTL. Must hold lock."""
        if self._ttl_seconds <= 0:
            return
        now = time.time()
        before = len(self._entries)
        self._entries = [
            e
            for e in self._entries
            if e.in_use or (now - e.last_used) <= self._ttl_seconds
        ]
        evicted = before - len(self._entries)
        if evicted > 0:
            logger.debug(f"Pool evicted {evicted} expired cache(s)")

    def _evict_oldest_idle(self) -> None:
        """Remove the oldest idle (not in_use) entry. Must hold lock."""
        oldest_idx = None
        oldest_time = float("inf")
        for i, entry in enumerate(self._entries):
            if not entry.in_use and entry.last_used < oldest_time:
                oldest_time = entry.last_used
                oldest_idx = i
        if oldest_idx is not None:
            removed = self._entries.pop(oldest_idx)
            logger.debug(
                f"Pool evicted idle cache (tokens: {len(removed.cache.tokens)})"
            )

    def get_pool_info(self) -> dict:
        """Get pool statistics for debugging/monitoring."""
        with self._lock:
            return {
                "pool_size": len(self._entries),
                "max_size": self._max_size,
                "in_use": sum(1 for e in self._entries if e.in_use),
                "idle": sum(1 for e in self._entries if not e.in_use),
                "ttl_seconds": self._ttl_seconds,
            }
