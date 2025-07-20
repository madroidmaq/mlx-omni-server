import re

import anthropic
import pytest
from fastapi.testclient import TestClient

from mlx_omni_server.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def anthropic_client(client):
    """Create Anthropic client configured with test server."""
    return anthropic.Anthropic(
        base_url="http://test/anthropic",
        api_key="not-needed",
        http_client=client,
    )


class TestAnthropicModels:
    def test_list_models(self, anthropic_client):
        """Test successful response from /anthropic/v1/models endpoint using SDK."""
        models_response = anthropic_client.models.list()

        assert models_response.data
        assert isinstance(models_response.data, list)
        assert models_response.first_id is not None
        assert models_response.last_id is not None
        assert models_response.has_more is not None

        if models_response.data:
            model = models_response.data[0]
            assert model.id
            assert model.display_name
            assert model.created_at
            assert model.type == "model"

            # Verify the created_at is a datetime object
            from datetime import datetime

            assert isinstance(model.created_at, datetime)

    def test_list_models_with_limit(self, anthropic_client):
        """Test the 'limit' query parameter using SDK."""
        # First, get all models to check if we can test the limit
        full_response = anthropic_client.models.list()

        if len(full_response.data) > 1:
            limited_response = anthropic_client.models.list(limit=1)
            assert len(limited_response.data) == 1
            assert limited_response.has_more is True
            assert limited_response.first_id == limited_response.data[0].id
            assert limited_response.last_id == limited_response.data[0].id
        else:
            # Not enough models to test limit, so we just check it returns what it has
            limited_response = anthropic_client.models.list(limit=1)
            assert len(limited_response.data) == len(full_response.data)
