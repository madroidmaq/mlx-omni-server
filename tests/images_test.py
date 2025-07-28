import logging

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from src.mlx_omni_server.main import app

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


def test_images(openai_client):
    """Test standard image generation functionality."""
    try:
        response = openai_client.images.generate(
            model="dhairyashil/FLUX.1-schnell-mflux-4bit",
            prompt="A cute baby sea otter.",
            n=1,
            size="512x512",
            response_format="url",
        )
        logger.info(f"Standard Image Response: {response}")

        assert response is not None, "No response received for standard generation"
        assert len(response.data) == 1, "Expected 1 image in the response data"
        assert response.data[0].url is not None, "Image URL should not be None"

    except Exception as e:
        logger.error(f"Test error in standard image generation: {str(e)}")
        raise


def test_images_b64_json(openai_client):
    """Test image generation with b64_json response format."""
    try:
        response = openai_client.images.generate(
            model="dhairyashil/FLUX.1-schnell-mflux-4bit",
            prompt="A beautiful sunset over the mountains.",
            n=1,
            size="512x512",
            response_format="b64_json",
        )
        logger.info("b64_json Image Response received.")

        assert response is not None, "No response received for b64_json generation"
        assert len(response.data) == 1, "Expected 1 image in the response data"
        assert response.data[0].b64_json is not None, "b64_json data should not be None"
        assert isinstance(
            response.data[0].b64_json, str
        ), "b64_json data should be a string"

    except Exception as e:
        logger.error(f"Test error in b64_json image generation: {str(e)}")
        raise
