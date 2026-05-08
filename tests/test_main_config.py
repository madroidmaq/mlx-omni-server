"""Tests for server startup configuration."""

from unittest.mock import patch

from mlx_omni_server import main


def test_default_model_cache_args_are_conservative(monkeypatch):
    """Default cache settings should favor lower memory usage."""
    monkeypatch.delenv("MLX_OMNI_MODEL_CACHE_SIZE", raising=False)
    monkeypatch.delenv("MLX_OMNI_MODEL_CACHE_TTL", raising=False)

    args = main.build_parser().parse_args([])

    assert args.model_cache_size == 1
    assert args.model_cache_ttl == 300


def test_model_cache_args_use_environment(monkeypatch):
    """Environment variables should configure parser defaults."""
    monkeypatch.setenv("MLX_OMNI_MODEL_CACHE_SIZE", "2")
    monkeypatch.setenv("MLX_OMNI_MODEL_CACHE_TTL", "60")

    args = main.build_parser().parse_args([])

    assert args.model_cache_size == 2
    assert args.model_cache_ttl == 60


def test_cli_model_cache_args_override_environment(monkeypatch):
    """CLI arguments should override environment-based defaults."""
    monkeypatch.setenv("MLX_OMNI_MODEL_CACHE_SIZE", "2")
    monkeypatch.setenv("MLX_OMNI_MODEL_CACHE_TTL", "60")

    args = main.build_parser().parse_args(
        ["--model-cache-size", "4", "--model-cache-ttl", "120"]
    )

    assert args.model_cache_size == 4
    assert args.model_cache_ttl == 120


def test_start_configures_wrapper_cache_from_cli(monkeypatch):
    """Server startup should apply parsed cache settings before uvicorn starts."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlx-omni-server",
            "--model-cache-size",
            "5",
            "--model-cache-ttl",
            "180",
        ],
    )

    with (
        patch.object(main.wrapper_cache, "configure") as mock_configure,
        patch.object(main.uvicorn, "run") as mock_run,
    ):
        main.start()

    mock_configure.assert_called_once_with(5, 180)
    mock_run.assert_called_once()


def test_start_exports_cache_config_for_workers(monkeypatch):
    """CLI cache settings should be exported so worker processes inherit them."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlx-omni-server",
            "--model-cache-size",
            "6",
            "--model-cache-ttl",
            "90",
        ],
    )

    with (
        patch.object(main.wrapper_cache, "configure"),
        patch.object(main.uvicorn, "run"),
    ):
        main.start()

    assert main.os.environ["MLX_OMNI_MODEL_CACHE_SIZE"] == "6"
    assert main.os.environ["MLX_OMNI_MODEL_CACHE_TTL"] == "90"


def test_start_warns_when_workers_greater_than_one(monkeypatch, caplog):
    """Multiple worker processes should warn about multiplied model memory."""
    monkeypatch.setattr("sys.argv", ["mlx-omni-server", "--workers", "2"])

    with (
        patch.object(main.wrapper_cache, "configure"),
        patch.object(main.uvicorn, "run") as mock_run,
        caplog.at_level("WARNING"),
    ):
        main.start()

    mock_run.assert_called_once()
    assert (
        "Each worker process maintains its own independent model cache" in caplog.text
    )
    assert "memory usage may scale up to the worker count" in caplog.text
