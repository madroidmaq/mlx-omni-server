import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from mlx_omni_server.chat.mlx.tools.reasoning_decoder import ReasoningDecoder
from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import logging

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


class TestReasoningResponse:
    """Test functionality of the ReasoningResponse class"""

    def test_reasoning_response(self, openai_client):
        """Test functionality of the ReasoningResponse class"""
        try:
            model = "mlx-community/Qwen3-0.6B-4bit"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
            )
            logger.info(f"Chat Completion Response:\n{response.choices[0].message}\n")

            # Validate response
            assert response.object == "chat.completion", "No usage in response"
            choices = response.choices[0]
            assert choices.message is not None, "No message in response"

            # 注意：在实际环境中，模型可能会在内容中包含</think>标签
            # 我们只需要验证响应中有内容，而不是检查特定标签的存在或不存在
            assert choices.message.content, "Message content is empty"

            # 从日志中可以看到，reasoning属性实际上是存在于message对象中，而不是choices[0]对象中
            assert (
                hasattr(choices.message, "reasoning")
                and choices.message.reasoning is not None
            ), "No reasoning in message"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_none_reasoning_response(self, openai_client):
        """Test functionality of the ReasoningResponse class"""
        try:
            model = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                extra_body={
                    "enable_thinking": False,
                },
            )
            logger.info(f"Chat Completion Response:\n{response.choices[0].message}\n")

            # Validate response
            assert response.object == "chat.completion", "No usage in response"
            choices = response.choices[0]
            assert choices.message is not None, "No message in response"
            assert (
                "</think>" in choices.message.content
            ), "Message content is not correct"
            assert (
                not hasattr(choices.message, "reasoning")
                or choices.message.reasoning is None
            ), "Has reasoning in message"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise


class TestReasoningDecoder:
    """Test functionality of the ReasoningDecoder class"""

    @pytest.fixture
    def tokenizer_mock(self):
        """Create a mock object for TokenizerWrapper"""
        tokenizer_mock = MagicMock()
        return tokenizer_mock

    @pytest.fixture
    def decoder(self, tokenizer_mock):
        """Create a ReasoningDecoder instance"""
        decoder = ReasoningDecoder(tokenizer_mock)
        return decoder

    def test_parse_response_with_thinking(self, decoder):
        """Test parsing responses with thinking tags"""
        # Prepare test data
        test_response = "<think>\nThis is a thinking process.\nAnalyzing the request.\n</think>\nHere is the final answer."

        # Execute test
        result = decoder._parse_response(test_response)

        # Verify results
        assert (
            result["reasoning"] == "This is a thinking process.\nAnalyzing the request."
        )
        assert result["content"] == "Here is the final answer."

    def test_parse_response_without_thinking(self, decoder):
        """Test parsing responses without thinking tags"""
        # Prepare test data
        test_response = "This is a direct response without thinking tags."

        # Execute test
        result = decoder._parse_response(test_response)

        # Verify results
        assert result["reasoning"] is None
        assert result["content"] == "This is a direct response without thinking tags."

    def test_decode_with_thinking_enabled(self, decoder):
        """Test decode method with thinking mode enabled"""
        # Prepare test data
        test_text = "<think>Reasoning process</think>Final answer"
        decoder.enable_thinking = True

        # Execute test
        result = decoder.decode(test_text)

        # Verify results
        assert result["reasoning"] == "Reasoning process"
        assert result["content"] == "Final answer"

    def test_decode_with_thinking_disabled(self, decoder):
        """Test decode method with thinking mode disabled"""
        # Prepare test data
        test_text = "<think>Reasoning process</think>Final answer"
        decoder.enable_thinking = False

        # Execute test
        result = decoder.decode(test_text)

        # Verify results
        assert result["content"] == test_text
        assert "reasoning" not in result

    def test_set_thinking_prefix(self, decoder):
        """Test the set_thinking_prefix method"""
        # Initial state
        assert decoder.accumulated_text == ""

        # Set to enabled
        decoder.set_thinking_prefix(True)
        assert decoder.add_thinking_prefix is True
        assert decoder.accumulated_text == f"<{decoder.thinking_tag}>"

        # Set to disabled
        decoder.set_thinking_prefix(False)
        assert decoder.add_thinking_prefix is False
        assert decoder.accumulated_text == ""

    def test_parse_stream_response_thinking_mode(self, decoder):
        """Test stream response parsing in thinking mode with sequential streaming calls"""
        # Reset state before test
        decoder.accumulated_text = ""

        # Step 1: Start with thinking tag
        result = decoder._parse_stream_response(f"<{decoder.thinking_tag}>")
        assert result["delta_content"] is None
        assert result["delta_reasoning"] == ""

        # Step 2: First part of thinking content
        result = decoder._parse_stream_response("I'm thinking ")
        assert result["delta_content"] is None
        assert result["delta_reasoning"] == "I'm thinking "

        # Step 3: Second part of thinking content
        result = decoder._parse_stream_response("about this problem.")
        assert result["delta_content"] is None
        assert result["delta_reasoning"] == "about this problem."

        # Step 4: Receive end tag
        result = decoder._parse_stream_response(f"</{decoder.thinking_tag}>")
        assert result["delta_content"] == ""
        assert result["delta_reasoning"] is None

        # Step 5: First part of final content
        result = decoder._parse_stream_response("Here is ")
        assert result["delta_content"] == "Here is "
        assert result["delta_reasoning"] is None

        # Step 6: Second part of final content
        result = decoder._parse_stream_response("the answer.")
        assert result["delta_content"] == "the answer."
        assert result["delta_reasoning"] is None

        # Verify accumulated text has everything
        expected_text = f"<{decoder.thinking_tag}>I'm thinking about this problem.</{decoder.thinking_tag}>Here is the answer."
        assert decoder.accumulated_text == expected_text

    def test_parse_response_missing_start_tag(self, decoder):
        """Test parsing responses with missing start tag but with end tag"""
        # Prepare test data that mimics the real-world example with missing start tag
        test_response = """Okay, the user is just greeting me.
</think>

Hello! How can I assist you today? 😊"""

        # Execute test
        result = decoder._parse_response(test_response)

        # Verify results
        expected_reasoning = """Okay, the user is just greeting me."""
        expected_content = "Hello! How can I assist you today? 😊"

        assert result["reasoning"] == expected_reasoning
        assert result["content"] == expected_content
