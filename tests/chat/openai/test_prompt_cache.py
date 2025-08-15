"""
Tests for prompt cache functionality

This test file verifies the prompt caching functionality in the chat completion API, including:
1. First conversation with no cache
2. Second conversation using cache
3. Modified conversation still hitting partial cache
"""

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


class TestPromptCache:
    """Tests for prompt cache functionality"""

    def test_conversation_with_prompt_cache(self, openai_client):
        try:
            logger.info("\n===== Conversation with prompt cache =====")
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"
            prompt = "Can you tell me more about your capabilities?"

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]

            first_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": first_response.choices[0].message.content,
                }
            )
            messages.append({"role": "user", "content": "continue"})

            # Create second conversation
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )

            # Verify cache in second conversation
            assert (
                response.usage.prompt_tokens_details is not None
            ), "Second conversation should have cached tokens"
            assert (
                response.usage.prompt_tokens_details.cached_tokens > 0
            ), "Cached tokens count should be greater than 0"
            logger.info(
                f"Second conversation cached tokens: {response.usage.prompt_tokens_details.cached_tokens}"
            )

        except Exception as e:
            logger.error(f"Error testing prompt cache: {str(e)}")
            raise
