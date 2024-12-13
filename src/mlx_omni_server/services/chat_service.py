from typing import Generator

from ..schemas.chat_schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from .chat.models import BaseMLXModel


class ChatService:
    model: BaseMLXModel = None

    def __init__(self, model: BaseMLXModel):
        self.model = model

    def generate_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        return self.model.generate(request)

    def generate_stream(
        self, request: ChatCompletionRequest
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Generate a streaming chat completion."""
        for chunk in self.model.stream_generate(request):
            yield chunk
