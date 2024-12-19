import argparse
import os

import uvicorn
from fastapi import FastAPI

from .middleware.logging import RequestResponseLoggingMiddleware
from .routers import api_router

app = FastAPI(title="MLX Omni Server")

# Add request/response logging middleware with custom levels
app.add_middleware(
    RequestResponseLoggingMiddleware,
    # exclude_paths=["/health"]
)

app.include_router(api_router)


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to, defaults to 10240",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of workers to use, defaults to 3",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level, defaults to info",
    )
    return parser


def start():
    """Start the MLX Omni Server."""
    parser = build_parser()
    args = parser.parse_args()

    # Set log level through environment variable
    os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level

    # Start server with uvicorn
    uvicorn.run(
        "mlx_omni_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )
