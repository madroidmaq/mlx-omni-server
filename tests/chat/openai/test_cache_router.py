from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_omni_server.chat.openai import cache_router


def _client():
    app = FastAPI()
    app.include_router(cache_router.router)
    return TestClient(app)


def test_get_cache_returns_wrapper_cache_info():
    client = _client()

    with patch("mlx_omni_server.chat.openai.cache_router.wrapper_cache") as mock_cache:
        mock_cache.get_cache_info.return_value = {
            "cache_size": 1,
            "max_size": 1,
            "ttl_seconds": 300,
            "cached_keys": ["model1"],
            "lru_order": ["model1"],
            "ttl_info": [],
        }

        response = client.get("/v1/cache")

    assert response.status_code == 200
    assert response.json()["cache_size"] == 1
    mock_cache.get_cache_info.assert_called_once_with()


def test_clear_cache_calls_wrapper_cache_clear_cache():
    client = _client()

    with patch("mlx_omni_server.chat.openai.cache_router.wrapper_cache") as mock_cache:
        mock_cache.clear_cache.return_value = 2
        mock_cache.get_cache_info.return_value = {
            "cache_size": 0,
            "max_size": 1,
            "ttl_seconds": 300,
            "cached_keys": [],
            "lru_order": [],
            "ttl_info": [],
        }

        response = client.post("/v1/cache/clear")

    assert response.status_code == 200
    assert response.json()["cleared_count"] == 2
    assert response.json()["cache_info"]["cache_size"] == 0
    mock_cache.clear_cache.assert_called_once_with()
