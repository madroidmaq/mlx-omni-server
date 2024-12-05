import json
import time
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logger import logger


def format_body(body: str) -> str:
    """Format body content for logging."""
    try:
        # Try to parse as JSON and format it
        parsed = json.loads(body)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        # If not JSON, return as is
        return body


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        exclude_paths: Optional[list[str]] = None,
    ):
        """Initialize the middleware with custom log levels.

        Args:
            app: The ASGI application
            request_level: Logging level for requests (default: INFO)
            response_level: Logging level for responses (default: INFO)
            exclude_paths: List of paths to exclude from logging (default: None)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or []

    def should_log(self, path: str) -> bool:
        """Check if the path should be logged."""
        return not any(path.startswith(exclude) for exclude in self.exclude_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.should_log(request.url.path):
            return await call_next(request)

        # Generate request ID for tracking
        request_id = str(time.time())

        # Log request
        body = await self._get_request_body(request)
        logger.debug(
            f"Request [{request_id}]: {request.method} {request.url}\n"
            f"Headers:\n{json.dumps(dict(request.headers), indent=2)}\n"
            f"Body:\n{format_body(body)}",
        )

        # Process the request and get response
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        if isinstance(response, StreamingResponse):
            logger.debug(
                f"Response [{request_id}] took {process_time:.2f}s:\n"
                f"Status: {response.status_code}\n"
                f"Headers:\n{json.dumps(dict(response.headers), indent=2)}\n"
                f"Body: <Streaming Response>",
            )
            return response
        else:
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            # Create a new async iterator for the response body
            async def body_iterator():
                yield response_body

            response.body_iterator = body_iterator()

            try:
                body_text = response_body.decode()
                body_text = format_body(body_text)
            except UnicodeDecodeError:
                body_text = "<Binary Content>"

            logger.debug(
                f"Response [{request_id}] took {process_time:.2f}s:\n"
                f"Status: {response.status_code}\n"
                f"Headers:\n{json.dumps(dict(response.headers), indent=2)}\n"
                f"Body:\n{body_text}",
            )
            return response

    async def _get_request_body(self, request: Request) -> str:
        """Get request body as string."""
        try:
            body = await request.body()
            return body.decode()
        except Exception:
            return ""
