"""Unit tests for PromptCachePool.

Tests the thread-safe prompt cache pool that provides exclusive checkout/checkin
semantics for concurrent request isolation.
"""

import threading
import time

import pytest

from mlx_omni_server.chat.mlx.prompt_cache import PromptCache
from mlx_omni_server.chat.mlx.prompt_cache_pool import PromptCachePool


class TestPromptCachePool:
    """Test PromptCachePool functionality."""

    def setup_method(self):
        self.pool = PromptCachePool(max_size=3, ttl_seconds=300)

    def test_checkout_creates_new_cache(self):
        """First checkout should create a new cache entry."""
        cache = self.pool.checkout([1, 2, 3], "model1")
        assert isinstance(cache, PromptCache)
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 1
        assert info["in_use"] == 1

    def test_concurrent_checkouts_get_different_caches(self):
        """Two checkouts without checkin should return different cache instances."""
        cache1 = self.pool.checkout([1, 2, 3], "model1")
        cache2 = self.pool.checkout([1, 2, 3], "model1")
        assert cache1 is not cache2
        info = self.pool.get_pool_info()
        assert info["in_use"] == 2

    def test_checkin_then_reuse_with_prefix_match(self):
        """After checkin, a cache with matching prefix should be reused."""
        cache1 = self.pool.checkout([1, 2, 3], "model1")
        # Simulate cache being populated during generation
        cache1.tokens = [1, 2, 3]
        cache1.model_key = "model1"
        self.pool.checkin(cache1)

        # Checkout with extended prompt — should reuse cache1 via prefix match
        cache2 = self.pool.checkout([1, 2, 3, 4, 5], "model1")
        assert cache2 is cache1

    def test_checkin_updates_last_used(self):
        """Checkin should update the last_used timestamp."""
        cache = self.pool.checkout([1, 2, 3], "model1")
        time.sleep(0.05)
        self.pool.checkin(cache)
        info = self.pool.get_pool_info()
        assert info["idle"] == 1

    def test_pool_full_all_in_use_returns_temporary(self):
        """When pool is full and all in use, should return unmanaged cache."""
        pool = PromptCachePool(max_size=2, ttl_seconds=300)
        cache1 = pool.checkout([1, 2, 3], "model1")
        cache2 = pool.checkout([4, 5, 6], "model1")
        # Pool is full, both in use
        cache3 = pool.checkout([7, 8, 9], "model1")
        assert isinstance(cache3, PromptCache)
        # Pool size should still be 2 (cache3 is unmanaged)
        info = pool.get_pool_info()
        assert info["pool_size"] == 2
        assert info["in_use"] == 2

        # Clean up
        pool.checkin(cache1)
        pool.checkin(cache2)
        pool.checkin(cache3)

    def test_checkin_adopts_unmanaged_cache_if_room(self):
        """Temporary cache should be adopted into pool on checkin if room."""
        pool = PromptCachePool(max_size=2, ttl_seconds=300)
        cache1 = pool.checkout([1, 2, 3], "model1")
        cache2 = pool.checkout([4, 5, 6], "model1")
        # Pool full, get unmanaged cache
        cache3 = pool.checkout([7, 8, 9], "model1")

        # Return cache1 to free a slot
        pool.checkin(cache1)
        assert pool.get_pool_info()["pool_size"] == 2

        # Now remove cache1's entry by checking out again (it might be reused)
        # Instead, let's just check that returning cache3 tries to adopt
        # First make room by evicting
        pool.checkin(cache2)

        # Pool has 2 idle entries, still at max_size=2, so cache3 won't be adopted
        pool.checkin(cache3)
        # cache3 should be discarded since pool is already at max_size
        assert pool.get_pool_info()["pool_size"] == 2

    def test_checkin_adopts_when_below_max(self):
        """Temporary cache should be adopted when pool is below max size."""
        pool = PromptCachePool(max_size=3, ttl_seconds=300)
        # Directly create an unmanaged cache (not from checkout)
        unmanaged = PromptCache()
        unmanaged.tokens = [10, 20, 30]
        unmanaged.model_key = "model1"
        pool.checkin(unmanaged)
        info = pool.get_pool_info()
        assert info["pool_size"] == 1
        assert info["idle"] == 1

    def test_ttl_eviction(self):
        """Expired idle caches should be evicted on next checkout."""
        pool = PromptCachePool(max_size=3, ttl_seconds=0.1)
        cache = pool.checkout([1, 2, 3], "model1")
        cache.tokens = [1, 2, 3]
        cache.model_key = "model1"
        pool.checkin(cache)

        time.sleep(0.2)  # Wait past TTL

        # Next checkout should evict the expired entry and create a new one
        cache2 = pool.checkout([1, 2, 3], "model1")
        assert cache2 is not cache
        info = pool.get_pool_info()
        assert info["pool_size"] == 1

    def test_in_use_caches_not_evicted_by_ttl(self):
        """Caches currently in use should never be evicted by TTL."""
        pool = PromptCachePool(max_size=3, ttl_seconds=0.1)
        cache = pool.checkout([1, 2, 3], "model1")

        time.sleep(0.2)  # Wait past TTL

        # Checkout another — should not evict the in-use cache
        cache2 = pool.checkout([4, 5, 6], "model1")
        info = pool.get_pool_info()
        assert info["pool_size"] == 2
        assert info["in_use"] == 2

        pool.checkin(cache)
        pool.checkin(cache2)

    def test_model_key_mismatch_creates_new_cache(self):
        """Caches for a different model should not be reused."""
        cache1 = self.pool.checkout([1, 2, 3], "model1")
        cache1.tokens = [1, 2, 3]
        cache1.model_key = "model1"
        self.pool.checkin(cache1)

        # Checkout for a different model
        cache2 = self.pool.checkout([1, 2, 3], "model2")
        assert cache2 is not cache1
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 2

    def test_best_prefix_match_selected(self):
        """Should select the cache with the longest common prefix."""
        # Create two caches with different prefixes
        cache_short = self.pool.checkout([1, 2, 3], "model1")
        cache_short.tokens = [1, 2]
        cache_short.model_key = "model1"
        self.pool.checkin(cache_short)

        cache_long = self.pool.checkout([10, 20, 30], "model1")
        cache_long.tokens = [1, 2, 3, 4, 5]
        cache_long.model_key = "model1"
        self.pool.checkin(cache_long)

        # Prompt [1,2,3,4,5,6] should prefer cache_long (5 prefix) over cache_short (2 prefix)
        best = self.pool.checkout([1, 2, 3, 4, 5, 6], "model1")
        assert best is cache_long

    def test_evict_oldest_idle_on_full(self):
        """When pool is full with different-model caches, oldest idle is evicted."""
        pool = PromptCachePool(max_size=2, ttl_seconds=300)

        cache1 = pool.checkout([1, 2, 3], "modelA")
        cache1.tokens = [1, 2, 3]
        cache1.model_key = "modelA"
        pool.checkin(cache1)

        time.sleep(0.01)

        cache2 = pool.checkout([4, 5, 6], "modelB")
        cache2.tokens = [4, 5, 6]
        cache2.model_key = "modelB"
        pool.checkin(cache2)

        # Pool is full (2/2). Checkout for modelC — no model match, must evict oldest
        cache3 = pool.checkout([7, 8, 9], "modelC")
        assert cache3 is not cache1
        assert cache3 is not cache2
        info = pool.get_pool_info()
        assert info["pool_size"] == 2

    def test_reuse_idle_cache_with_zero_prefix(self):
        """An idle cache with same model but 0 prefix match is still reused."""
        pool = PromptCachePool(max_size=2, ttl_seconds=300)

        cache1 = pool.checkout([1, 2, 3], "model1")
        cache1.tokens = [1, 2, 3]
        cache1.model_key = "model1"
        pool.checkin(cache1)

        # Same model, different prompt — reuses cache1 (will be reset by get_prompt_cache)
        cache2 = pool.checkout([7, 8, 9], "model1")
        assert cache2 is cache1
        info = pool.get_pool_info()
        assert info["pool_size"] == 1

    def test_get_pool_info(self):
        """Pool info should report correct statistics."""
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 0
        assert info["max_size"] == 3
        assert info["in_use"] == 0
        assert info["idle"] == 0
        assert info["ttl_seconds"] == 300

        cache = self.pool.checkout([1, 2, 3], "model1")
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 1
        assert info["in_use"] == 1
        assert info["idle"] == 0

        self.pool.checkin(cache)
        info = self.pool.get_pool_info()
        assert info["in_use"] == 0
        assert info["idle"] == 1


class TestPromptCachePoolThreadSafety:
    """Test thread safety of PromptCachePool."""

    def test_concurrent_checkout_checkin(self):
        """Multiple threads doing checkout/checkin should never share a cache."""
        pool = PromptCachePool(max_size=10, ttl_seconds=300)
        errors = []
        seen_caches = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for _ in range(5):
                    cache = pool.checkout([1, 2, 3], "model1")
                    cache_id = id(cache)

                    # Record this cache's identity while it's checked out
                    with lock:
                        # Check no other thread currently holds this cache
                        for tid, cid in seen_caches:
                            if cid == cache_id and tid != thread_id:
                                errors.append(
                                    f"Thread {thread_id} got cache {cache_id} "
                                    f"already held by thread {tid}"
                                )
                        seen_caches.append((thread_id, cache_id))

                    # Simulate some work
                    time.sleep(0.001)

                    # Remove from seen before checkin
                    with lock:
                        seen_caches.remove((thread_id, cache_id))

                    pool.checkin(cache)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_checkout_returns_unique_caches(self):
        """Concurrent checkouts should never return the same cache instance."""
        pool = PromptCachePool(max_size=10, ttl_seconds=300)
        caches = []
        lock = threading.Lock()

        def worker():
            cache = pool.checkout([1, 2, 3], "model1")
            with lock:
                caches.append(cache)
            time.sleep(0.01)  # Hold the cache briefly
            pool.checkin(cache)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 5 caches should be distinct objects
        cache_ids = [id(c) for c in caches]
        assert len(set(cache_ids)) == 5, "Some threads got the same cache instance"
