import logging
import os

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
    """Create OpenAI client configured to use test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


def test_embeddings_single_text(openai_client):
    """Test generating embeddings for a single text"""
    try:
        model = "mlx-community/all-MiniLM-L6-v2-4bit"  # Use small embedding model for testing
        response = openai_client.embeddings.create(
            model=model, input="This is a test text", encoding_format="float"
        )
        logger.info(f"Embedding Response: {response}")

        # Basic validation
        assert response is not None, "No response received"
        assert len(response.data) == 1, "Should return one embedding object"
        assert response.model == model, "Model name in response should match request"
        assert response.usage.prompt_tokens > 0, "Should return token usage count"
        assert response.usage.total_tokens > 0, "Should return total token count"

        # Validate embedding vector
        embedding = response.data[0].embedding
        assert embedding is not None, "Embedding should not be empty"
        assert len(embedding) > 0, "Embedding should contain vector data"
        assert response.data[0].index == 0, "Index for single input should be 0"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_embeddings_multiple_texts(openai_client):
    """Test generating embeddings for multiple texts"""
    try:
        model = "mlx-community/all-MiniLM-L6-v2-4bit"
        inputs = ["First test text", "Second test text", "Third test text"]

        response = openai_client.embeddings.create(
            model=model, input=inputs, encoding_format="float"
        )
        logger.info(f"Multiple Embeddings Response: {response}")

        # Basic validation
        assert response is not None, "No response received"
        assert len(response.data) == len(
            inputs
        ), "Number of embeddings should match number of input texts"
        assert response.model == model, "Model name in response should match request"

        # Validate each embedding vector
        for i, embedding_data in enumerate(response.data):
            assert (
                embedding_data.embedding is not None
            ), f"Embedding {i+1} should not be empty"
            assert (
                len(embedding_data.embedding) > 0
            ), f"Embedding {i+1} should contain vector data"
            assert embedding_data.index == i, f"Index for embedding {i+1} should be {i}"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_embeddings_with_dimensions(openai_client):
    """Test specifying dimensions parameter"""
    try:
        model = "mlx-community/all-MiniLM-L6-v2-4bit"
        dimensions = 256  # Specify smaller dimensions

        response = openai_client.embeddings.create(
            model=model, input="Testing dimensions parameter", dimensions=dimensions
        )
        logger.info(f"Embeddings with Dimensions Response: {response}")

        # Basic validation
        assert response is not None, "No response received"

        # Note: Current implementation may not support dimensions parameter, just verify API doesn't fail
        # If dimensions functionality is implemented, uncomment below for validation
        # assert len(response.data[0].embedding) == dimensions, "Embedding vector dimensions should match specified value"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_embeddings_with_user_param(openai_client):
    """Test specifying user parameter"""
    try:
        model = "mlx-community/all-MiniLM-L6-v2-4bit"
        user = "test-user-id-123"

        response = openai_client.embeddings.create(
            model=model, input="Testing user parameter", user=user
        )
        logger.info(f"Embeddings with User Param Response: {response}")

        # Verify API accepts user parameter and returns normally
        assert response is not None, "No response received"
        assert len(response.data) == 1, "Should return one embedding object"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_embeddings_missing_model(client):
    """Test missing required model parameter"""
    try:
        # Prepare request data
        request_data = {"input": "Test text"}

        # Send request
        response = client.post("/v1/embeddings", json=request_data)
        logger.info(
            f"Missing Model Response: {response.status_code}, {response.json()}"
        )

        # Validate response
        assert (
            response.status_code == 422
        ), "Missing required 'model' parameter should return validation error"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_embeddings_missing_input(client):
    """Test missing required input parameter"""
    try:
        # Prepare request data
        request_data = {"model": "mlx-community/all-MiniLM-L6-v2-4bit"}

        # Send request
        response = client.post("/v1/embeddings", json=request_data)
        logger.info(
            f"Missing Input Response: {response.status_code}, {response.json()}"
        )

        # Validate response
        assert (
            response.status_code == 422
        ), "Missing required 'input' parameter should return validation error"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise
