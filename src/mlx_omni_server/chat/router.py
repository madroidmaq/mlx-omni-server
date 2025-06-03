import json
from dataclasses import dataclass
from typing import Generator, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .mlx.models import ModelId, load_model
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel

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
        for chunk in text_model.stream_generate(request):
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
) -> BaseTextModel:
    """Create a text model based on the model parameters.

    Creates a ModelId object and passes it to load_model function.
    The caching is handled inside the load_model function.
    """
    current_key = ModelId(
        model_id=model_id, adapter_path=adapter_path, draft_model=draft_model
    )

    return load_model(current_key)
