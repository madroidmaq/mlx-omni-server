from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, TypedDict

from .schema import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse


class GenerationParams(TypedDict):
    sampler_kwargs: Dict[str, Any]
    model_kwargs: Dict[str, Any]
    generate_kwargs: Dict[str, Any]
    template_kwargs: Dict[str, Any]


@dataclass
class GenerateResult:
    """Result from generate step"""

    text: str
    token: int
    finish_reason: Optional[str]
    prompt_tokens: int
    generation_tokens: int
    logprobs: Optional[Dict[str, Any]] = None


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
