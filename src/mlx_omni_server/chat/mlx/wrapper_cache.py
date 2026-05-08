"""Centralized ChatGenerator Cache

This module provides a unified caching system for ChatGenerator instances
to avoid expensive model reloading when the same model configuration is used
across different API endpoints.
"""

import gc
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...utils.logger import logger
from .chat_generator import ChatGenerator

DEFAULT_MODEL_CACHE_SIZE = 1
DEFAULT_MODEL_CACHE_TTL_SECONDS = 300
MODEL_CACHE_SIZE_ENV = "MLX_OMNI_MODEL_CACHE_SIZE"
MODEL_CACHE_TTL_ENV = "MLX_OMNI_MODEL_CACHE_TTL"


def parse_non_negative_int(value: str | int, name: str) -> int:
    """Parse a non-negative integer config value."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc

    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def get_model_cache_config_from_env(
    environ: dict[str, str] | None = None
) -> tuple[int, int]:
    """Read model cache configuration from environment variables."""
    environ = os.environ if environ is None else environ
    max_size = parse_non_negative_int(
        environ.get(MODEL_CACHE_SIZE_ENV, DEFAULT_MODEL_CACHE_SIZE),
        MODEL_CACHE_SIZE_ENV,
    )
    ttl_seconds = parse_non_negative_int(
        environ.get(MODEL_CACHE_TTL_ENV, DEFAULT_MODEL_CACHE_TTL_SECONDS),
        MODEL_CACHE_TTL_ENV,
    )
    return max_size, ttl_seconds


def clear_mlx_memory() -> None:
    """Run best-effort Python and MLX memory cleanup."""
    gc.collect()

    try:
        import mlx.metal as metal
    except ModuleNotFoundError:
        logger.debug("mlx.metal is not available; skipped MLX cache cleanup")
        return
    except Exception as e:
        logger.warning(f"Failed to import mlx.metal for cache cleanup: {e}")
        return

    try:
        clear_cache = getattr(metal, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
    except Exception as e:
        logger.warning(f"Failed to clear MLX metal cache: {e}")


@dataclass(frozen=True)
class WrapperCacheKey:
    """Cache key for ChatGenerator instances.

    Uses all parameters that affect model loading to ensure proper cache invalidation
    when any of these parameters change.
    """

    model_id: str
    adapter_path: Optional[str] = None
    draft_model_id: Optional[str] = None


class MLXWrapperCache:
    """Thread-safe LRU cache for ChatGenerator instances with TTL support.

    This cache ensures that expensive model loading only happens once per unique
    combination of (model_id, adapter_path, draft_model_id). All API endpoints
    (OpenAI, Anthropic) can share the same cached wrapper instance.

    Uses LRU (Least Recently Used) eviction policy and TTL (Time To Live)
    to manage memory usage automatically.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MODEL_CACHE_SIZE,
        ttl_seconds: int = DEFAULT_MODEL_CACHE_TTL_SECONDS,
        cleanup_interval: int = 5,
    ):
        """Initialize cache with LRU eviction and TTL support.

        Args:
            max_size: Maximum number of models to cache.
            ttl_seconds: Time to live in seconds, after which unused models
                        are evicted from cache.
            cleanup_interval: Interval in seconds for background cleanup.
        """
        self._cache: OrderedDict[WrapperCacheKey, ChatGenerator] = OrderedDict()
        self._access_times: Dict[WrapperCacheKey, float] = {}
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._max_size = parse_non_negative_int(max_size, "max_size")
        self._ttl_seconds = parse_non_negative_int(ttl_seconds, "ttl_seconds")
        self._cleanup_interval = cleanup_interval
        self._stop_event = threading.Event()
        self._cleanup_thread = None

        if self._ttl_seconds > 0:
            self._cleanup_thread = threading.Thread(
                target=self._periodic_cleanup, daemon=True
            )
            self._cleanup_thread.start()

    def _release_evicted_items(
        self, evicted_items: list[tuple[WrapperCacheKey, ChatGenerator]]
    ) -> None:
        """Release cache-owned references after wrappers leave the cache."""
        if not evicted_items:
            return
        evicted_items.clear()
        clear_mlx_memory()

    def _pop_lru_locked(self) -> tuple[WrapperCacheKey, ChatGenerator] | None:
        if not self._cache:
            return None
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        wrapper = self._cache.pop(lru_key, None)
        self._access_times.pop(lru_key, None)
        logger.info(f"Evicted LRU model from cache: {lru_key}")
        if wrapper is None:
            return None
        return lru_key, wrapper

    def _evict_expired_items_locked(
        self,
    ) -> list[tuple[WrapperCacheKey, ChatGenerator]]:
        """Evict items that have exceeded their TTL while holding the lock."""
        if self._ttl_seconds <= 0:
            return []

        current_time = time.time()
        expired_keys = []

        for key, access_time in self._access_times.items():
            if current_time - access_time > self._ttl_seconds:
                expired_keys.append(key)

        evicted_items = []
        for key in expired_keys:
            wrapper = self._cache.pop(key, None)
            self._access_times.pop(key, None)
            if wrapper is not None:
                evicted_items.append((key, wrapper))
            logger.info(
                f"Evicted expired model from cache (TTL={self._ttl_seconds}s): {key}"
            )

        return evicted_items

    def _evict_lru_for_new_item_locked(
        self,
    ) -> list[tuple[WrapperCacheKey, ChatGenerator]]:
        """Evict one LRU item if needed before adding a new cache entry."""
        if self._max_size <= 0 or len(self._cache) < self._max_size:
            return []
        item = self._pop_lru_locked()
        return [item] if item is not None else []

    def _shrink_to_max_size_locked(self) -> list[tuple[WrapperCacheKey, ChatGenerator]]:
        """Evict cached entries until cache_size <= max_size."""
        evicted_items = []
        while len(self._cache) > self._max_size:
            item = self._pop_lru_locked()
            if item is None:
                break
            evicted_items.append(item)
        return evicted_items

    def _update_access_time(self, key: WrapperCacheKey) -> None:
        """Update access time for LRU tracking.

        This method should be called while holding the lock.
        """
        self._access_times[key] = time.time()

    def _periodic_cleanup(self) -> None:
        """Background thread method for periodic cleanup of expired items.

        This method runs in a daemon thread and periodically checks for expired items.
        """
        while not self._stop_event.wait(self._cleanup_interval):
            try:
                with self._lock:
                    evicted_items = self._evict_expired_items_locked()
                self._release_evicted_items(evicted_items)
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    def _stop_cleanup_thread(self) -> None:
        """Stop the background cleanup thread gracefully."""
        if self._cleanup_thread is not None:
            self._stop_event.set()
            self._cleanup_thread.join(timeout=1.0)
            self._cleanup_thread = None
            logger.info("Background cleanup thread stopped")

    def _get_cached_wrapper(self, key: WrapperCacheKey) -> ChatGenerator | None:
        with self._lock:
            evicted_items = self._evict_expired_items_locked()
            wrapper = self._cache.get(key)
            if wrapper is not None:
                self._update_access_time(key)
                logger.debug(f"Cache hit for ChatGenerator: {key}")
        self._release_evicted_items(evicted_items)
        return wrapper

    def get_wrapper(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        draft_model_id: Optional[str] = None,
    ) -> ChatGenerator:
        """Get or create ChatGenerator instance.

        Args:
            model_id: Model name/path (HuggingFace model ID or local path)
            adapter_path: Optional path to LoRA adapter
            draft_model_id: Optional draft model name/path for speculative decoding

        Returns:
            Cached or newly created ChatGenerator instance

        Note:
            This method is thread-safe and will only create one wrapper instance
            per unique parameter combination, even under concurrent access.
        """
        key = WrapperCacheKey(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )

        wrapper = self._get_cached_wrapper(key)
        if wrapper is not None:
            return wrapper

        with self._load_lock:
            wrapper = self._get_cached_wrapper(key)
            if wrapper is not None:
                return wrapper

            with self._lock:
                evicted_items = self._evict_expired_items_locked()
                if key in self._cache:
                    self._update_access_time(key)
                    wrapper = self._cache[key]
                else:
                    evicted_items.extend(self._evict_lru_for_new_item_locked())
                    wrapper = None
            self._release_evicted_items(evicted_items)

            if wrapper is not None:
                logger.debug(f"Cache hit (after load lock) for ChatGenerator: {key}")
                return wrapper

            logger.info(f"Creating new ChatGenerator for: {key}")
            try:
                wrapper = ChatGenerator.create(
                    model_id=model_id,
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                )
            except Exception as e:
                logger.error(f"Failed to create ChatGenerator for {key}: {e}")
                raise

            with self._lock:
                evicted_items = []
                if self._max_size > 0:
                    evicted_items = self._evict_lru_for_new_item_locked()
                    self._cache[key] = wrapper
                    self._update_access_time(key)
                    logger.info(
                        f"Successfully cached ChatGenerator: {key} (cache size: {len(self._cache)}/{self._max_size})"
                    )
                else:
                    logger.info(
                        f"Created ChatGenerator but not cached (max_size=0): {key}"
                    )
            self._release_evicted_items(evicted_items)
            return wrapper

    def cleanup_expired_items(self) -> int:
        """Manually trigger cleanup of expired items.

        This can be called periodically by a background task or manually
        to clean up expired items without waiting for cache access.

        Returns:
            Number of items that were evicted
        """
        with self._lock:
            evicted_items = self._evict_expired_items_locked()
        evicted_count = len(evicted_items)
        self._release_evicted_items(evicted_items)

        if evicted_count > 0:
            logger.info(f"Manual cleanup evicted {evicted_count} expired items")

        return evicted_count

    def clear_cache(self) -> int:
        """Clear all cached wrapper instances.

        This can be useful for memory management or testing purposes.

        Returns:
            Number of cached wrappers that were cleared.
        """
        with self._lock:
            items = list(self._cache.items())
            self._cache.clear()
            self._access_times.clear()

        cache_size = len(items)
        self._release_evicted_items(items)
        logger.info(f"Cleared ChatGenerator cache ({cache_size} entries)")
        return cache_size

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including LRU and TTL information
        """
        with self._lock:
            evicted_items = self._evict_expired_items_locked()

            current_time = time.time()
            sorted_keys = sorted(
                self._access_times.items(), key=lambda x: x[1], reverse=True
            )

            ttl_info = []
            if self._ttl_seconds > 0:
                for key, access_time in sorted_keys:
                    remaining_ttl = self._ttl_seconds - (current_time - access_time)
                    ttl_info.append(
                        {
                            "key": str(key),
                            "remaining_ttl_seconds": max(0, remaining_ttl),
                            "expires_at": access_time + self._ttl_seconds,
                        }
                    )

            cache_info = {
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "cached_keys": [str(key) for key in self._cache.keys()],
                "lru_order": [str(key) for key, _ in sorted_keys],
                "ttl_info": ttl_info,
            }

        self._release_evicted_items(evicted_items)
        return cache_info

    def set_max_size(self, max_size: int) -> None:
        """Update the maximum cache size.

        Args:
            max_size: New maximum cache size

        Note:
            If the new size is smaller than current cache size,
            LRU items will be evicted immediately.
        """
        max_size = parse_non_negative_int(max_size, "max_size")
        with self._lock:
            self._max_size = max_size
            evicted_items = self._shrink_to_max_size_locked()
            current_size = len(self._cache)
        self._release_evicted_items(evicted_items)
        logger.info(
            f"Updated cache max_size to {max_size}, current size: {current_size}"
        )

    def set_ttl_seconds(self, ttl_seconds: int) -> None:
        """Update cache TTL seconds."""
        ttl_seconds = parse_non_negative_int(ttl_seconds, "ttl_seconds")
        stop_cleanup_thread = False
        with self._lock:
            self._ttl_seconds = ttl_seconds

            if self._ttl_seconds > 0 and self._cleanup_thread is None:
                self._stop_event.clear()
                self._cleanup_thread = threading.Thread(
                    target=self._periodic_cleanup, daemon=True
                )
                self._cleanup_thread.start()
            elif self._ttl_seconds <= 0:
                stop_cleanup_thread = True

            logger.info(f"Updated cache ttl_seconds to {ttl_seconds}")

        if stop_cleanup_thread:
            self._stop_cleanup_thread()

    def configure(self, max_size: int, ttl_seconds: int) -> None:
        """Update cache max size and TTL."""
        self.set_max_size(max_size)
        self.set_ttl_seconds(ttl_seconds)

    def __del__(self) -> None:
        """Destructor to ensure cleanup thread is stopped."""
        self._stop_cleanup_thread()


wrapper_cache = MLXWrapperCache(*get_model_cache_config_from_env())
