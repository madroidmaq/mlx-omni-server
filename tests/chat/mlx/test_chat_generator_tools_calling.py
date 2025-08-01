"""Tests for ChatGenerator tools calling."""

import logging

import pytest

from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator
from mlx_omni_server.chat.mlx.core_types import CompletionContent, GenerationResult

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def model_cache():
    """Create a model cache with gemma-3-1b for testing."""
    model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    return ChatGenerator.create(model_name).model


@pytest.fixture
def mlx_wrapper():
    """Create ChatGenerator instance for testing."""
    model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    return ChatGenerator.create(model_name)


class TestChatGeneratorToolsCalling:
    """Test ChatGenerator tools calling functionality."""

    def test_reasoning_tools_calling(self):
        model_name = "mlx-community/Qwen3-0.6B-4bit-DWQ"
        mlx_wrapper = ChatGenerator.create(model_name)

        messages = [{"role": "user", "content": "What's the weather like in New York?"}]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        result = mlx_wrapper.generate(
            messages=messages,
            tools=tools,
            template_kwargs={"enable_thinking": True},
        )

        print("\n============================\nResult:", result)
        logger.info(f"Generation result: {result}")
        logger.info(f"Tool calls: {result.content.tool_calls}")
        logger.info(f"Reasoning: {result.content.reasoning}")
        logger.info(f"Text: {result.content.text}")
        logger.info(f"Stats: {result.stats}")
        logger.info(
            f"Token usage: prompt={result.stats.prompt_tokens}, completion={result.stats.completion_tokens}"
        )
        logger.info(
            f"Performance: prompt_tps={result.stats.prompt_tps}, generation_tps={result.stats.generation_tps}"
        )

        assert isinstance(result, GenerationResult)
        assert isinstance(result.content, CompletionContent)
        assert result.content.reasoning is not None
        assert result.content.tool_calls is not None
