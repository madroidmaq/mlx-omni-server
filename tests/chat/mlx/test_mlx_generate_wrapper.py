"""Tests for MLXGenerateWrapper."""

import pytest

from mlx_omni_server.chat.mlx.core_types import GenerationResult, ToolCall
from mlx_omni_server.chat.mlx.mlx_generate_wrapper import MLXGenerateWrapper
from mlx_omni_server.chat.mlx.model_types import MlxModelCache, ModelId


@pytest.fixture
def model_cache():
    """Create a model cache with gemma-3-1b for testing."""
    model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    return MlxModelCache(model_id=ModelId(name=model_name))


@pytest.fixture
def mlx_wrapper(model_cache):
    """Create MLXGenerateWrapper instance for testing."""
    return MLXGenerateWrapper(model_cache)


class TestMLXGenerateWrapper:
    """Test MLXGenerateWrapper functionality."""

    def test_initialization(self, mlx_wrapper):
        """Test basic initialization."""
        assert mlx_wrapper.model_cache is not None
        assert mlx_wrapper.tokenizer is not None
        assert mlx_wrapper.chat_template is not None
        assert mlx_wrapper._prompt_cache is None

    def test_basic_generate(self, mlx_wrapper):
        """Test basic text generation."""
        messages = [{"role": "user", "content": "Say hello"}]

        result = mlx_wrapper.generate(messages=messages, max_tokens=10)

        assert isinstance(result, GenerationResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.stats.prompt_tokens > 0
        assert result.stats.completion_tokens > 0
        # finish_reason might be None if we hit max_tokens or other conditions

    def test_generate_with_temperature(self, mlx_wrapper):
        """Test generation with different temperature settings."""
        messages = [{"role": "user", "content": "Count to 3"}]

        # Test with low temperature (more deterministic)
        result_low = mlx_wrapper.generate(
            messages=messages, temperature=0.1, max_tokens=20
        )

        # Test with high temperature (more random)
        result_high = mlx_wrapper.generate(
            messages=messages, temperature=1.5, max_tokens=20
        )

        assert isinstance(result_low, GenerationResult)
        assert isinstance(result_high, GenerationResult)
        assert len(result_low.text) > 0
        assert len(result_high.text) > 0

    def test_generate_with_sampler_config(self, mlx_wrapper):
        """Test generation with custom sampler configuration."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        sampler_kwargs = {"min_p": 0.1, "xtc_probability": 0.1}

        result = mlx_wrapper.generate(
            messages=messages,
            max_tokens=20,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            sampler_kwargs=sampler_kwargs,
        )

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0

    def test_generate_with_mlx_config(self, mlx_wrapper):
        """Test generation with MLX-specific configuration."""
        messages = [{"role": "user", "content": "Explain gravity briefly"}]

        result = mlx_wrapper.generate(
            messages=messages,
            max_tokens=50,  # Function parameter takes precedence
            repetition_penalty=1.1,
            max_kv_size=2048,  # Valid parameter for generate_step
        )

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.stats.completion_tokens <= 50

    def test_generate_with_logprobs(self, mlx_wrapper):
        """Test generation with logprobs enabled."""
        messages = [{"role": "user", "content": "Hi"}]

        result = mlx_wrapper.generate(messages=messages, max_tokens=5, top_logprobs=3)

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        # Note: logprobs might be None if not supported by model

    def test_stream_generate(self, mlx_wrapper):
        """Test streaming generation."""
        messages = [{"role": "user", "content": "Count from 1 to 5"}]

        results = []
        for result in mlx_wrapper.stream_generate(messages=messages, max_tokens=30):
            results.append(result)
            assert isinstance(result, GenerationResult)
            assert isinstance(result.text, str)

        assert len(results) > 0

        # Combine all text chunks
        full_text = "".join(result.text for result in results)
        assert len(full_text) > 0

    def test_stream_generate_with_configs(self, mlx_wrapper):
        """Test streaming generation with custom configurations."""
        messages = [{"role": "user", "content": "Tell me about cats"}]

        sampler_kwargs = {"min_p": 0.1, "xtc_probability": 0.1}

        results = []
        for result in mlx_wrapper.stream_generate(
            messages=messages,
            max_tokens=40,  # Function parameter takes precedence
            temperature=0.7,
            top_p=0.95,
            sampler_kwargs=sampler_kwargs,
        ):
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        assert final_result.stats.completion_tokens <= 40

    def test_prompt_caching(self, mlx_wrapper):
        """Test prompt caching functionality."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]

        # First generation - should populate cache
        result1 = mlx_wrapper.generate(
            messages=messages, max_tokens=20, enable_prompt_cache=True
        )

        # Second generation with same prompt - should use cache
        result2 = mlx_wrapper.generate(
            messages=messages, max_tokens=20, enable_prompt_cache=True
        )

        assert isinstance(result1, GenerationResult)
        assert isinstance(result2, GenerationResult)
        assert len(result1.text) > 0
        assert len(result2.text) > 0

    def test_disable_prompt_caching(self, mlx_wrapper):
        """Test disabling prompt caching."""
        messages = [{"role": "user", "content": "Test without cache"}]

        result = mlx_wrapper.generate(
            messages=messages, max_tokens=15, enable_prompt_cache=False
        )

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.stats.cache_hit_tokens == 0

    def test_error_handling_empty_messages(self, mlx_wrapper):
        """Test error handling with empty messages."""
        with pytest.raises(Exception):
            mlx_wrapper.generate(messages=[])

    def test_error_handling_invalid_max_tokens(self, mlx_wrapper):
        """Test error handling with invalid max_tokens."""
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(Exception):
            mlx_wrapper.generate(messages=messages, max_tokens=0)

    def test_multiple_message_conversation(self, mlx_wrapper):
        """Test generation with multi-turn conversation."""
        messages = [
            {"role": "user", "content": "What is 5 + 3?"},
            {"role": "assistant", "content": "5 + 3 = 8"},
            {"role": "user", "content": "What about 8 * 2?"},
        ]

        result = mlx_wrapper.generate(messages=messages, max_tokens=20)

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0

    def test_parameter_precedence(self, mlx_wrapper):
        """Test that function parameters take precedence."""
        messages = [{"role": "user", "content": "Test precedence"}]

        # Function parameter should be respected
        result = mlx_wrapper.generate(messages=messages, max_tokens=10)

        assert isinstance(result, GenerationResult)
        assert result.stats.completion_tokens <= 10  # Should respect function parameter

    def test_streaming_reasoning_mode(self):
        """Test streaming with reasoning/thinking enabled."""
        reasoning_model = MlxModelCache(
            model_id=ModelId(name="mlx-community/Qwen3-0.6B-4bit")
        )
        reasoning_wrapper = MLXGenerateWrapper(reasoning_model)

        messages = [{"role": "user", "content": "Calculate 23 * 17"}]

        template_kwargs = {"enable_thinking": True}

        results = []
        for result in reasoning_wrapper.stream_generate(
            messages=messages, max_tokens=80, template_kwargs=template_kwargs
        ):
            results.append(result)

        assert len(results) > 0
        final_text = "".join(result.text for result in results)
        assert len(final_text) > 0

    def test_reasoning_model_qwen(self):
        """Test with actual reasoning model (Qwen3-0.6B-4bit) - optional test."""
        try:
            # Try to load reasoning model
            reasoning_model = MlxModelCache(
                model_id=ModelId(name="mlx-community/Qwen3-0.6B-4bit")
            )
            reasoning_wrapper = MLXGenerateWrapper(reasoning_model)

            messages = [
                {
                    "role": "user",
                    "content": "Think about this step by step: What is 5 + 3?",
                }
            ]
            template_kwargs = {"enable_thinking": True}

            result = reasoning_wrapper.generate(
                messages=messages, template_kwargs=template_kwargs
            )

            assert isinstance(result, GenerationResult)
            assert len(result.text) > 0
        except Exception as e:
            import pytest

            pytest.skip(f"Reasoning model not available: {e}")

    def test_generate_with_empty_tools(self, mlx_wrapper):
        """Test generation with empty tools list."""
        messages = [{"role": "user", "content": "Hello"}]

        result = mlx_wrapper.generate(messages=messages, tools=[], max_tokens=10)

        assert isinstance(result, GenerationResult)
        assert result.tool_calls is None
        assert len(result.text) > 0

    def test_generate_with_none_values(self, mlx_wrapper):
        """Test generation with None values for optional parameters."""
        messages = [{"role": "user", "content": "Test"}]

        result = mlx_wrapper.generate(
            messages=messages,
            tools=None,
            sampler_kwargs=None,
            template_kwargs=None,
            max_tokens=10,
        )

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
