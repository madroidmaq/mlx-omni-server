from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from ...schemas.chat_schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


@dataclass
class GenerateResult:
    """Result from generate step"""

    text: str
    token: int
    finish_reason: Optional[str]
    prompt_tokens: int
    generation_tokens: int
    logprobs: Optional[Dict[str, Any]] = None


class BaseMLXModel(ABC):
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
