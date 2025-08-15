import json
from typing import Generator, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator
from mlx_omni_server.chat.openai.openai_adapter import OpenAIAdapter
from mlx_omni_server.chat.openai.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

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

    Uses the shared wrapper cache to get or create ChatGenerator instance.
    This avoids expensive model reloading when the same model configuration
    is used across different requests or API endpoints.
    """
    # Get cached or create new ChatGenerator
    wrapper = ChatGenerator.get_or_create(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model,
    )

    # Create OpenAIAdapter with the cached wrapper directly
    return OpenAIAdapter(wrapper=wrapper)


# Legacy caching variables removed - now using shared wrapper_cache
# This eliminates duplicate caching logic and enables sharing between endpoints
