"""Core data types for MLX generation wrapper."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar


@dataclass
class ToolCall:
    """Platform-independent tool call representation."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class GenerationStats:
    """Statistics for generation performance and token usage."""

    # Token counting
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Caching statistics
    cache_hit_tokens: int = 0  # Number of tokens served from cache

    # Performance metrics
    prompt_tps: float = 0.0  # Tokens per second for prompt processing
    generation_tps: float = 0.0  # Tokens per second for generation
    peak_memory: float = 0.0  # Peak memory usage in MB


@dataclass
class ChatTemplateResult:
    content: str
    thinking: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


# ========== Generic Content Types ==========

ContentT = TypeVar("ContentT", bound="BaseContent")


@dataclass
class BaseContent:
    """Content type base class."""

    pass


@dataclass
class GenerationResult(Generic[ContentT]):
    """Unified generation result container - generics ensure type safety.

    This is the main result container that uses generics to provide type safety
    for different content types (streaming vs completion).
    """

    content: ContentT  # Core: content type determined by generic
    finish_reason: Optional[str] = None
    stats: GenerationStats = field(default_factory=GenerationStats)
    logprobs: Optional[Dict[str, Any]] = None
    from_draft: bool = False

    # Legacy fields for backward compatibility - deprecated, use content instead
    text: str = field(default="", init=False)
    token: int = field(default=0, init=False)
    tool_calls: Optional[List[ToolCall]] = field(default=None, init=False)
    reasoning: Optional[str] = field(default=None, init=False)
    raw_delta: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Set legacy fields for backward compatibility."""
        if hasattr(self.content, "text"):
            self.text = self.content.text
        elif hasattr(self.content, "text_delta") and self.content.text_delta:
            self.text = self.content.text_delta

        if hasattr(self.content, "token"):
            self.token = self.content.token

        if hasattr(self.content, "tool_calls"):
            self.tool_calls = self.content.tool_calls

        if hasattr(self.content, "reasoning"):
            self.reasoning = self.content.reasoning
        elif hasattr(self.content, "reasoning_delta") and self.content.reasoning_delta:
            self.reasoning = self.content.reasoning_delta

        # Set raw_delta for streaming content
        if hasattr(self.content, "text_delta") and self.content.text_delta:
            self.raw_delta = self.content.text_delta
        elif hasattr(self.content, "reasoning_delta") and self.content.reasoning_delta:
            self.raw_delta = self.content.reasoning_delta


@dataclass
class StreamContent(BaseContent):
    """Stream incremental content - semantically clear incremental data."""

    # Core incremental fields
    text_delta: Optional[str] = None  # Normal text increment
    reasoning_delta: Optional[str] = None  # Thinking process increment
    token: int = 0  # Current generated token

    # Stream-specific fields
    chunk_index: int = 0  # Incremental sequence index

    def __post_init__(self):
        """Data consistency validation."""
        active_deltas = sum(
            x is not None for x in [self.text_delta, self.reasoning_delta]
        )
        if active_deltas != 1:
            raise ValueError("Exactly one delta field must be non-None")


@dataclass
class CompletionContent(BaseContent):
    """Completion content - clear final result semantics."""

    # Core content fields
    text: str = ""  # Complete generated text
    reasoning: Optional[str] = None  # Complete thinking process
    tool_calls: Optional[List[ToolCall]] = None  # Complete tool calls

    # Token information - semantically clear separated design
    text_tokens: List[int] = field(default_factory=list)  # Normal text token sequence
    reasoning_tokens: Optional[List[int]] = None  # Thinking process token sequence


# Type aliases to improve user experience
StreamResult = GenerationResult[StreamContent]
CompletionResult = GenerationResult[CompletionContent]
