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

            assert choices.message.content, "Message content is empty"

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
            model = "mlx-community/Qwen3-0.6B-4bit"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                extra_body={
                    "enable_thinking": False,
                },
            )
            logger.info(
                f"=============== Chat Completion Response ===============:\n{response.choices[0].message}\n"
            )

            # Validate response
            assert response.object == "chat.completion", "No usage in response"
            choices = response.choices[0]
            assert choices.message is not None, "No message in response"
            assert (
                "</think>" not in choices.message.content
            ), "Message content is not correct"
            assert choices.message.reasoning is None, "Has reasoning in message"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise
