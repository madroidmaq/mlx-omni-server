import json
from typing import Generator, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omni_server.chat.anthropic.anthropic_messages_adapter import (
    AnthropicMessagesAdapter,
)

from ..mlx.model_types import MLXModel
from .anthropic_schema import MessagesRequest, MessagesResponse
from .models_service import AnthropicModelsService
from .schema import AnthropicModelList

router = APIRouter(tags=["anthropic"])
models_service = AnthropicModelsService()

# Initialize global cache objects
_cached_model: MLXModel = None
_cached_anthropic_adapter: AnthropicMessagesAdapter = None


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
    return models_service.list_models(
        limit=limit, after_id=after_id, before_id=before_id
    )


@router.post("/messages", response_model=MessagesResponse)
@router.post("/v1/messages", response_model=MessagesResponse)
async def create_message(request: MessagesRequest):
    """Create an Anthropic Messages API completion"""

    anthropic_model = _create_anthropic_model(
        request.model,
        # Extract extra params if needed - for now use defaults
        None,  # adapter_path
        None,  # draft_model
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
) -> AnthropicMessagesAdapter:
    """Create an Anthropic Messages adapter based on the model parameters.

    Creates a MLXModel object and passes it to load_anthropic_adapter function.
    The caching is handled inside the load_anthropic_adapter function.
    """
    current_key = MLXModel.load(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model,
    )

    return _load_anthropic_adapter(current_key)


def _load_anthropic_adapter(model_key: MLXModel) -> AnthropicMessagesAdapter:
    """Load the model and return an AnthropicMessagesAdapter instance.

    Args:
        model_key: MLXModel object containing model identification parameters

    Returns:
        Initialized AnthropicMessagesAdapter instance
    """
    global _cached_model, _cached_anthropic_adapter

    # Check if a new model needs to be loaded
    model_needs_reload = _cached_model is None or _cached_model != model_key

    if model_needs_reload:
        # Cache miss, use the already loaded model
        _cached_model = model_key

        # Create and cache new AnthropicMessagesAdapter instance
        _cached_anthropic_adapter = AnthropicMessagesAdapter(model=_cached_model)
    elif _cached_anthropic_adapter is None:
        # Model is cached but anthropic adapter is not
        _cached_anthropic_adapter = AnthropicMessagesAdapter(model=_cached_model)

    # Return cached adapter instance
    return _cached_anthropic_adapter
