import logging

import pytest
from fastapi.testclient import TestClient
from openai import NotFoundError, OpenAI

from src.mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with the test server."""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


def test_list_models_default(openai_client: OpenAI):
    """Test listing models without details (default)."""
    model_list = openai_client.models.list()
    if not model_list.data:
        pytest.skip("No models available to test.")
    for model in model_list.data:
        assert not hasattr(model, "details") or model.details is None


def test_list_models_with_details(openai_client: OpenAI):
    """Test listing models with the show_details flag."""
    model_list = openai_client.models.list(
        extra_query={
            "include_details": True,
        }
    )
    if not model_list.data:
        pytest.skip("No models available to test.")
    for model in model_list.data:
        assert model.details is not None
        assert isinstance(model.details, dict)


def test_get_existing_model_with_details(openai_client: OpenAI):
    """Test retrieving a single, existing model with details."""
    # First, get a valid model ID from the list
    model_list = openai_client.models.list()
    if not model_list.data:
        pytest.skip("No models available in the cache to test retrieval.")

    model_id_to_test = model_list.data[0].id

    try:
        model = openai_client.models.retrieve(
            model_id_to_test,
            extra_query={
                "include_details": True,
            },
        )
        logger.info(f"Retrieved Model with details: {model}")

        assert model is not None
        assert model.id == model_id_to_test
        assert model.details is not None
        assert isinstance(model.details, dict)
        assert model.details.get("model_type") is not None

    except Exception as e:
        logger.error(
            f"Test error retrieving model '{model_id_to_test}' with details: {str(e)}"
        )
        raise


def test_get_existing_model_without_details(openai_client: OpenAI):
    """Test retrieving a single, existing model without details."""
    # First, get a valid model ID from the list
    model_list = openai_client.models.list()
    if not model_list.data:
        pytest.skip("No models available in the cache to test retrieval.")

    model_id_to_test = model_list.data[0].id

    try:
        model = openai_client.models.retrieve(
            model_id_to_test,
            extra_query={
                "include_details": False,
            },
        )
        logger.info(f"Retrieved Model without details: {model}")

        assert model is not None
        assert model.id == model_id_to_test
        assert not hasattr(model, "details") or model.details is None

    except Exception as e:
        logger.error(
            f"Test error retrieving model '{model_id_to_test}' without details: {str(e)}"
        )
        raise


def test_get_non_existent_model(openai_client: OpenAI):
    """Test retrieving a non-existent model."""
    non_existent_model_id = "non-existent/model-that-should-not-be-found"
    with pytest.raises(NotFoundError):
        openai_client.models.retrieve(non_existent_model_id)
