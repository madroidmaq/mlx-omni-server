from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, TypedDict

from .schema import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse


class GenerationParams(TypedDict):
    sampler_kwargs: Dict[str, Any]
    model_kwargs: Dict[str, Any]
    generate_kwargs: Dict[str, Any]
    template_kwargs: Dict[str, Any]


class BaseTextModel(ABC):
    """Base class for chat models"""

    @abstractmethod
    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate completion text with parameters from request"""
        pass

    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        pass
