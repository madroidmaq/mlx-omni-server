import json
import os

from mlx_omni_server.chat.mlx.exo_chat_generator import get_exo_generator
from typing import Generator, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omni_server.chat.anthropic.anthropic_messages_adapter import (
    AnthropicMessagesAdapter,
)

from ..mlx.chat_generator import ChatGenerator
from .anthropic_schema import MessagesRequest, MessagesResponse
from .models_service import AnthropicModelsService
from .schema import AnthropicModelList

router = APIRouter(tags=["anthropic"])

# Lazy initialization to avoid scanning cache during module import
_models_service = None


def get_models_service() -> AnthropicModelsService:
    """Get or create the anthropic models service singleton with lazy initialization."""
    global _models_service
    if _models_service is None:
        _models_service = AnthropicModelsService()
    return _models_service


# Legacy caching variables removed - now using shared wrapper_cache
# This eliminates duplicate caching logic and enables sharing between endpoints


@router.get("/models", response_model=AnthropicModelList)
@router.get("/v1/models", response_model=AnthropicModelList)
async def list_anthropic_models(
    before_id: Optional[str] = Query(
        default=None,
        title="Before Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately before this object.",
    ),
    after_id: Optional[str] = Query(
        default=None,
        title="After Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately after this object.",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=1000,
        title="Limit",
        description="Number of items to return per page. Defaults to 20. Ranges from 1 to 1000.",
    ),
) -> AnthropicModelList:
    """List available models in Anthropic format."""
    return get_models_service().list_models(
        limit=limit, after_id=after_id, before_id=before_id
    )


@router.post("/messages", response_model=MessagesResponse)
@router.post("/v1/messages", response_model=MessagesResponse)
async def create_message(request: MessagesRequest):
    """Create an Anthropic Messages API completion"""

    model_id = os.environ.get("MLX_OMNI_MODEL") or request.model
    exo = get_exo_generator(model_id)
    anthropic_model = _create_anthropic_model(
        model_id,
        # Extract extra params if needed - for now use defaults
        None,  # adapter_path
        None,  # draft_model
        exo_generator=exo,
    )

    if not request.stream:
        completion = anthropic_model.generate(request)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    async def anthropic_event_generator() -> Generator[str, None, None]:
        for event in anthropic_model.generate_stream(request):
            yield f"event: {event.type.value}\n"
            yield f"data: {json.dumps(event.model_dump(exclude_none=True))}\n\n"

    return StreamingResponse(
        anthropic_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _create_anthropic_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model: Optional[str] = None,
    exo_generator=None,
) -> AnthropicMessagesAdapter:
    """Create an Anthropic Messages adapter based on the model parameters.

    Uses the shared wrapper cache to get or create ChatGenerator instance.
    This avoids expensive model reloading when the same model configuration
    is used across different requests or API endpoints.
    """
    # Use Exo cluster if available, otherwise load model locally
    if exo_generator is not None:
        return AnthropicMessagesAdapter(wrapper=exo_generator)

    # Get cached or create new ChatGenerator
    wrapper = ChatGenerator.get_or_create(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model,
    )

    # Create AnthropicMessagesAdapter with the cached wrapper directly
    return AnthropicMessagesAdapter(wrapper=wrapper)
