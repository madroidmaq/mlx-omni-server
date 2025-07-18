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


@dataclass
class SamplerConfig:
    """Configuration for MLX sampler (passed to make_sampler)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None  # None means no top_k filtering
    min_p: float = 0.0
    min_tokens_to_keep: int = 1

    # Extended sampling parameters
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0


@dataclass
class MLXGenerateConfig:
    """Configuration for MLX generation (passed to mlx_lm.generate)."""

    max_tokens: int = 2048

    # Performance options
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    kv_group_size: int = 64
    quantized_kv_start: int = 0

    # Draft model for speculative decoding
    num_draft_tokens: int = 3

    # Generation control
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None

    # JSON schema for structured output
    json_schema: Optional[Dict[str, Any]] = None

    # Logprobs configuration (MLX generate function parameter)
    top_logprobs: Optional[int] = (
        None  # None means no logprobs, int > 0 enables logprobs
    )


@dataclass
class ChatTemplateConfig:
    """Configuration for chat template (passed to tokenizer.apply_chat_template)."""

    # Reasoning/thinking parameters
    enable_thinking: bool = True
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[str] = None

    # Template-specific parameters
    template_kwargs: Dict[str, Any] = field(default_factory=dict)
