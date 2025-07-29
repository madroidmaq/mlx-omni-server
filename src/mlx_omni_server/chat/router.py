import json
from typing import Generator, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .mlx.model_types import MLXModel
from .openai_adapter import OpenAIAdapter
from .schema import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter(tags=["chat—completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""

    text_model = _create_text_model(
        request.model,
        request.get_extra_params().get("adapter_path"),
        request.get_extra_params().get("draft_model"),
    )

    if not request.stream:
        completion = text_model.generate(request)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    async def event_generator() -> Generator[str, None, None]:
        for chunk in text_model.generate_stream(request):
            yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _create_text_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model: Optional[str] = None,
) -> OpenAIAdapter:
    """Create a text model based on the model parameters.

    Creates a MLXModel object and passes it to load_model function.
    The caching is handled inside the load_model function.
    """
    current_key = MLXModel.load(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model,
    )

    return _load_openai_adapter(current_key)


_cached_model: MLXModel = None
_cached_adapter: OpenAIAdapter = None


def _load_openai_adapter(model_key: MLXModel) -> OpenAIAdapter:
    """Load the model and return an OpenAIAdapter instance.

    Args:
        model_key: MLXModel object containing model identification parameters

    Returns:
        Initialized OpenAIAdapter instance
    """
    global _cached_model, _cached_adapter

    # Check if a new model needs to be loaded
    model_needs_reload = _cached_model is None or _cached_model != model_key

    if model_needs_reload:
        # Cache miss, use the already loaded model
        _cached_model = model_key

        # Create and cache new OpenAIAdapter instance
        _cached_adapter = OpenAIAdapter(model=_cached_model)

    # Return cached adapter instance
    return _cached_adapter
