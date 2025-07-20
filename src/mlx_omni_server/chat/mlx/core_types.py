"""Core data types for MLX generation wrapper."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


@dataclass
class GenerationResult:
    """Unified generation result format for all API platforms.

    This structure contains all possible fields that different API platforms
    might need, allowing adapters to select relevant fields.
    """

    # Core generation fields
    text: str
    token: int
    finish_reason: Optional[str]

    # Generation statistics
    stats: GenerationStats

    # Extended functionality fields
    tool_calls: Optional[List[ToolCall]] = None
    reasoning: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None

    # Draft model info (for speculative decoding)
    from_draft: bool = False

    # Raw delta for non-streaming reconstruction
    raw_delta: Optional[str] = None
