"""Tests for MLXGenerateWrapper."""

import logging

import pytest

from mlx_omni_server.chat.mlx.core_types import GenerationResult, ToolCall
from mlx_omni_server.chat.mlx.mlx_generate_wrapper import MLXGenerateWrapper
from mlx_omni_server.chat.mlx.model_types import MlxModelCache, ModelId

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def model_cache():
    """Create a model cache with gemma-3-1b for testing."""
    model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    return MlxModelCache(model_id=ModelId(name=model_name))


@pytest.fixture
def mlx_wrapper(model_cache):
    """Create MLXGenerateWrapper instance for testing."""
    return MLXGenerateWrapper(model_cache)


class TestMLXGenerateWrapperToolsCalling:
    """Test MLXGenerateWrapper functionality."""

    def test_tools_calling(self, mlx_wrapper):
        """Test tools/function calling functionality."""
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

        result = mlx_wrapper.generate(messages=messages, tools=tools)
        print("\n============================\nResult:", result)
        logger.info(f"Generation result: {result}")
        logger.info(f"Tool calls: {result.tool_calls}")
        logger.info(f"Reasoning: {result.reasoning}")
        logger.info(f"Text: {result.text}")
        logger.info(f"Stats: {result.stats}")
        logger.info(
            f"Token usage: prompt={result.stats.prompt_tokens}, completion={result.stats.completion_tokens}"
        )
        logger.info(
            f"Performance: prompt_tps={result.stats.prompt_tps}, generation_tps={result.stats.generation_tps}"
        )
