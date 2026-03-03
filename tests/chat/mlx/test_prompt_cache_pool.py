"""Unit tests for PromptCachePool.

Tests the deepcopy-based prompt cache pool that gives each request its own
independent PromptCache while reusing prefix-matched state from prior requests.
"""

import time

from mlx_omni_server.chat.mlx.prompt_cache import PromptCache
from mlx_omni_server.chat.mlx.prompt_cache_pool import PromptCachePool


class TestPromptCachePool:
    """Test PromptCachePool functionality."""

    def setup_method(self):
        self.pool = PromptCachePool(max_size=3, ttl_seconds=300)

    def test_get_cache_creates_new_entry(self):
        """First get_cache should create a new cache entry in the pool."""
        cache = self.pool.get_cache([1, 2, 3], "model1")
        assert isinstance(cache, PromptCache)
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 1

    def test_get_cache_returns_deepcopy_on_match(self):
        """get_cache should return a deepcopy, not the original pool entry."""
        # Seed the pool with a cache
        pool = PromptCachePool(max_size=3, ttl_seconds=300)
        cache1 = pool.get_cache([1, 2, 3], "model1")
        cache1.tokens = [1, 2, 3]
        cache1.model_key = "model1"
        pool.put_cache(cache1)

        # get_cache with matching prefix returns a deepcopy
        cache2 = pool.get_cache([1, 2, 3, 4, 5], "model1")
        assert cache2 is not cache1
        # Deepcopy should have the same token content
        assert cache2.tokens == [1, 2, 3]
        assert cache2.model_key == "model1"

    def test_deepcopy_isolates_mutations(self):
        """Mutating a returned cache should not affect the pool entry."""
        pool = PromptCachePool(max_size=3, ttl_seconds=300)
        cache1 = pool.get_cache([1, 2, 3], "model1")
        cache1.tokens = [1, 2, 3]
        cache1.model_key = "model1"
        pool.put_cache(cache1)

        # Get a copy and mutate it
        cache2 = pool.get_cache([1, 2, 3], "model1")
        cache2.tokens = [99, 99, 99]

        # Pool entry should be unaffected
        cache3 = pool.get_cache([1, 2, 3], "model1")
        assert cache3.tokens == [1, 2, 3]

    def test_put_cache_adds_entry(self):
        """put_cache should add a new entry to the pool."""
        cache = PromptCache()
        cache.tokens = [10, 20, 30]
        cache.model_key = "model1"
        self.pool.put_cache(cache)
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 1

    def test_put_cache_evicts_oldest_when_over_max(self):
        """put_cache should evict oldest entry when pool exceeds max_size."""
        pool = PromptCachePool(max_size=2, ttl_seconds=300)

        cache_a = PromptCache()
        cache_a.tokens = [1, 2, 3]
        cache_a.model_key = "model1"
        pool.put_cache(cache_a)

        time.sleep(0.01)

        cache_b = PromptCache()
        cache_b.tokens = [4, 5, 6]
        cache_b.model_key = "model1"
        pool.put_cache(cache_b)

        time.sleep(0.01)

        # Adding a third should evict the oldest (cache_a)
        cache_c = PromptCache()
        cache_c.tokens = [7, 8, 9]
        cache_c.model_key = "model1"
        pool.put_cache(cache_c)

        info = pool.get_pool_info()
        assert info["pool_size"] == 2

        # The remaining caches should be cache_b and cache_c (cache_a evicted)
        result = pool.get_cache([4, 5, 6], "model1")
        assert result.tokens == [4, 5, 6]

    def test_best_prefix_match_selected(self):
        """get_cache should select the cache with the longest common prefix."""
        pool = PromptCachePool(max_size=3, ttl_seconds=300)

        cache_short = PromptCache()
        cache_short.tokens = [1, 2]
        cache_short.model_key = "model1"
        pool.put_cache(cache_short)

        cache_long = PromptCache()
        cache_long.tokens = [1, 2, 3, 4, 5]
        cache_long.model_key = "model1"
        pool.put_cache(cache_long)

        # Prompt [1,2,3,4,5,6] should prefer cache_long (5 prefix) over cache_short (2 prefix)
        best = pool.get_cache([1, 2, 3, 4, 5, 6], "model1")
        assert best.tokens == [1, 2, 3, 4, 5]

    def test_model_key_mismatch_creates_new_cache(self):
        """Caches for a different model should not be matched."""
        cache1 = PromptCache()
        cache1.tokens = [1, 2, 3]
        cache1.model_key = "model1"
        self.pool.put_cache(cache1)

        # get_cache for a different model should create a new empty cache
        cache2 = self.pool.get_cache([1, 2, 3], "model2")
        assert cache2.tokens == []
        assert cache2.model_key == ""
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 2

    def test_ttl_eviction(self):
        """Expired caches should be evicted on next get_cache."""
        pool = PromptCachePool(max_size=3, ttl_seconds=0.1)

        cache = PromptCache()
        cache.tokens = [1, 2, 3]
        cache.model_key = "model1"
        pool.put_cache(cache)

        time.sleep(0.2)  # Wait past TTL

        # Next get_cache should evict the expired entry and create a new one
        cache2 = pool.get_cache([1, 2, 3], "model1")
        assert cache2.tokens == []  # Fresh cache, not the expired one
        info = pool.get_pool_info()
        assert info["pool_size"] == 1

    def test_ttl_disabled(self):
        """With ttl_seconds=0, no entries should be evicted by TTL."""
        pool = PromptCachePool(max_size=3, ttl_seconds=0)

        cache = PromptCache()
        cache.tokens = [1, 2, 3]
        cache.model_key = "model1"
        pool.put_cache(cache)

        time.sleep(0.05)

        result = pool.get_cache([1, 2, 3], "model1")
        assert result.tokens == [1, 2, 3]

    def test_get_cache_evicts_oldest_when_full_no_match(self):
        """When pool is full and no model matches, oldest entry is evicted."""
        pool = PromptCachePool(max_size=2, ttl_seconds=300)

        cache_a = PromptCache()
        cache_a.tokens = [1, 2, 3]
        cache_a.model_key = "modelA"
        pool.put_cache(cache_a)

        time.sleep(0.01)

        cache_b = PromptCache()
        cache_b.tokens = [4, 5, 6]
        cache_b.model_key = "modelB"
        pool.put_cache(cache_b)

        # Pool full. get_cache for modelC evicts oldest (cache_a) and creates new
        cache_c = pool.get_cache([7, 8, 9], "modelC")
        assert cache_c.tokens == []
        info = pool.get_pool_info()
        assert info["pool_size"] == 2

    def test_zero_prefix_match_still_returns_deepcopy(self):
        """A cache with same model but 0 prefix match still gets returned."""
        pool = PromptCachePool(max_size=2, ttl_seconds=300)

        cache = PromptCache()
        cache.tokens = [1, 2, 3]
        cache.model_key = "model1"
        pool.put_cache(cache)

        # Same model, completely different prompt — still matches (prefix_len=0 > -1)
        result = pool.get_cache([7, 8, 9], "model1")
        assert result.tokens == [1, 2, 3]
        assert result is not cache  # deepcopy

    def test_get_pool_info(self):
        """Pool info should report correct statistics."""
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 0
        assert info["max_size"] == 3
        assert info["utilization"] == 0.0
        assert info["ttl_seconds"] == 300

        cache = PromptCache()
        cache.tokens = [1, 2, 3]
        cache.model_key = "model1"
        self.pool.put_cache(cache)
        info = self.pool.get_pool_info()
        assert info["pool_size"] == 1
        assert info["utilization"] == 1 / 3

    def test_full_lifecycle_get_and_put(self):
        """Simulate a complete request lifecycle: get_cache, use, put_cache."""
        pool = PromptCachePool(max_size=3, ttl_seconds=300)

        # Request 1: get a fresh cache, simulate generation, put it back
        cache1 = pool.get_cache([1, 2, 3], "model1")
        cache1.tokens = [1, 2, 3, 10, 11]  # prompt + generated tokens
        cache1.model_key = "model1"
        pool.put_cache(cache1)

        # Request 2: same conversation extended — should get prefix match
        cache2 = pool.get_cache([1, 2, 3, 10, 11, 4, 5], "model1")
        assert cache2.tokens == [1, 2, 3, 10, 11]
        assert cache2 is not cache1
