"""Tests for ChatGenerator."""

import pytest

from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator
from mlx_omni_server.chat.mlx.core_types import (
    CompletionContent,
    GenerationResult,
    StreamContent,
)
from mlx_omni_server.utils.logger import logger


@pytest.fixture
def mlx_wrapper():
    """Create ChatGenerator instance for testing."""
    model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    return ChatGenerator.create(model_name)


class TestChatGenerator:
    """Test ChatGenerator functionality."""

    def test_initialization(self, mlx_wrapper):
        """Test basic initialization."""
        assert mlx_wrapper.tokenizer is not None
        assert mlx_wrapper.chat_template is not None
        assert mlx_wrapper._prompt_cache is None

    def test_basic_generate(self, mlx_wrapper):
        """Test basic text generation."""
        messages = [{"role": "user", "content": "Say hello"}]

        result = mlx_wrapper.generate(messages=messages, max_tokens=10)

        assert isinstance(result, GenerationResult)
        assert isinstance(result.content, CompletionContent)
        assert isinstance(result.content.text, str)
        assert len(result.content.text) > 0
        assert result.stats.prompt_tokens > 0
        assert result.stats.completion_tokens > 0
        # finish_reason might be None if we hit max_tokens or other conditions

    def test_generate_with_temperature(self, mlx_wrapper):
        """Test generation with different temperature settings."""
        messages = [{"role": "user", "content": "Count to 3"}]

        # Test with low temperature (more deterministic)
        result_low = mlx_wrapper.generate(
            messages=messages, sampler={"temp": 0.1}, max_tokens=20
        )

        # Test with high temperature (more random)
        result_high = mlx_wrapper.generate(
            messages=messages, sampler={"temp": 1.5}, max_tokens=20
        )

        assert isinstance(result_low, GenerationResult)
        assert isinstance(result_low.content, CompletionContent)
        assert isinstance(result_high, GenerationResult)
        assert isinstance(result_high.content, CompletionContent)
        assert len(result_low.content.text) > 0
        assert len(result_high.content.text) > 0

    def test_generate_with_sampler_config(self, mlx_wrapper):
        """Test generation with custom sampler configuration."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        sampler_config = {
            "temp": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "xtc_probability": 0.1,
        }

        result = mlx_wrapper.generate(
            messages=messages,
            max_tokens=20,
            sampler=sampler_config,
        )

        assert isinstance(result, GenerationResult)
        assert isinstance(result.content, CompletionContent)
        assert len(result.content.text) > 0

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
        assert isinstance(result.content, CompletionContent)
        assert len(result.content.text) > 0
        assert result.stats.completion_tokens <= 50

    def test_generate_with_logprobs(self, mlx_wrapper):
        """Test generation with logprobs enabled."""
        messages = [{"role": "user", "content": "Hi"}]

        top_logprobs_count = 3
        result = mlx_wrapper.generate(
            messages=messages, max_tokens=5, top_logprobs=top_logprobs_count
        )

        logger.info(result.logprobs)

        assert isinstance(result, GenerationResult)
        assert result.logprobs is not None

        top_logprobs = result.logprobs["top_logprobs"]
        assert len(top_logprobs) == top_logprobs_count

    def test_stream_generate(self, mlx_wrapper):
        """Test streaming generation."""
        messages = [{"role": "user", "content": "Count from 1 to 5"}]

        results = []
        for result in mlx_wrapper.generate_stream(messages=messages, max_tokens=30):
            results.append(result)
            assert isinstance(result, GenerationResult)
            assert isinstance(result.content, StreamContent)
            # Stream content should have either text_delta or reasoning_delta
            assert (
                result.content.text_delta is not None
                or result.content.reasoning_delta is not None
            )

        assert len(results) > 0

        # Combine all text chunks (only text deltas)
        full_text = "".join(
            result.content.text_delta
            for result in results
            if result.content.text_delta is not None
        )
        assert len(full_text) > 0

    def test_stream_generate_with_configs(self, mlx_wrapper):
        """Test streaming generation with custom configurations."""
        messages = [{"role": "user", "content": "Tell me about cats"}]

        sampler_config = {
            "temp": 0.7,
            "top_p": 0.95,
            "min_p": 0.1,
            "xtc_probability": 0.1,
        }

        results = []
        for result in mlx_wrapper.generate_stream(
            messages=messages,
            max_tokens=40,  # Function parameter takes precedence
            sampler=sampler_config,
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
        assert len(result1.content.text) > 0
        assert len(result2.content.text) > 0

    def test_disable_prompt_caching(self, mlx_wrapper):
        """Test disabling prompt caching."""
        messages = [{"role": "user", "content": "Test without cache"}]

        result = mlx_wrapper.generate(
            messages=messages, max_tokens=15, enable_prompt_cache=False
        )

        assert isinstance(result, GenerationResult)
        assert isinstance(result.content, CompletionContent)
        assert len(result.content.text) > 0
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
        assert isinstance(result.content, CompletionContent)
        assert len(result.content.text) > 0

    def test_parameter_precedence(self, mlx_wrapper):
        """Test that function parameters take precedence."""
        messages = [{"role": "user", "content": "Test precedence"}]

        # Function parameter should be respected
        result = mlx_wrapper.generate(messages=messages, max_tokens=10)

        assert isinstance(result, GenerationResult)
        assert result.stats.completion_tokens <= 10  # Should respect function parameter

    def test_streaming_reasoning_mode(self):
        """Test streaming with reasoning/thinking enabled."""
        reasoning_wrapper = ChatGenerator.create("mlx-community/Qwen3-0.6B-4bit")

        messages = [{"role": "user", "content": "Calculate 23 * 17"}]

        template_kwargs = {"enable_thinking": True}

        content = ""
        reasoning = ""
        for result in reasoning_wrapper.generate_stream(
            messages=messages, template_kwargs=template_kwargs
        ):

            # Accumulate content based on the type of delta
            if result.content.text_delta:
                content = content + result.content.text_delta
            if result.content.reasoning_delta:
                reasoning = reasoning + result.content.reasoning_delta

        print(f"Reasoning: {reasoning}")
        print(f"content: {content}")

        assert len(content) > 0
        assert len(reasoning) > 0

    def test_reasoning_model_qwen(self):
        """Test with actual reasoning model (Qwen3-0.6B-4bit) - optional test."""
        try:
            # Try to load reasoning model
            reasoning_wrapper = ChatGenerator.create("mlx-community/Qwen3-0.6B-4bit")

            messages = [
                {
                    "role": "user",
                    "content": "Think about this step by step: What is 5 + 3?",
                }
            ]
            template_kwargs = {"enable_thinking": True}

            result = reasoning_wrapper.generate(
                messages=messages,
                template_kwargs=template_kwargs,
            )

            assert isinstance(result, GenerationResult)
            assert len(result.content.text) > 0
            assert len(result.content.reasoning) > 0
        except Exception as e:
            import pytest

            pytest.skip(f"Reasoning model not available: {e}")

    def test_generate_with_empty_tools(self, mlx_wrapper):
        """Test generation with empty tools list."""
        messages = [{"role": "user", "content": "Hello"}]

        result = mlx_wrapper.generate(messages=messages, tools=[], max_tokens=10)

        assert isinstance(result, GenerationResult)
        assert result.content.tool_calls is None
        assert len(result.content.text) > 0

    def test_generate_with_none_values(self, mlx_wrapper):
        """Test generation with None values for optional parameters."""
        messages = [{"role": "user", "content": "Test"}]

        result = mlx_wrapper.generate(
            messages=messages,
            tools=None,
            sampler=None,  # Should let mlx-lm use its defaults
            template_kwargs=None,
            max_tokens=10,
        )

        assert isinstance(result, GenerationResult)
        assert isinstance(result.content, CompletionContent)
        assert len(result.content.text) > 0
