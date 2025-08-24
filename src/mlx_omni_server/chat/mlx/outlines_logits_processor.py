from typing import Any, Dict, Type, Union

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from outlines.models.transformers import TransformerTokenizer
from outlines.processors import JSONLogitsProcessor
from outlines.processors.structured import RegexLogitsProcessor
from outlines.types import JsonSchema
from outlines_core.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel


class OutlinesLogitsProcessor:
    processed_token_count: int = 0

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        schema: Union[Dict[str, Any], Type[BaseModel], str],
        enable_thinking: bool = False,
    ):
        """Initialize the OutlinesLogitsProcessor.

        Args:
            tokenizer: MLX tokenizer wrapper
            schema: JSON schema dictionary, Pydantic BaseModel class, or JSON schema string
            enable_thinking: Whether to enable thinking pattern support
        """
        self.enable_thinking = enable_thinking
        self.schema = schema

        if enable_thinking:
            # Complex but precise thinking pattern to avoid premature matching
            # This pattern carefully excludes matching </think> until it's complete
            thinking_pattern: str = (
                r"<think>([^<]|<[^\/]|<\/[^t]|<\/t[^h]|<\/th[^i]|<\/thi[^n]|<\/thin[^k]|<\/think[^>])*<\/think>\n"
            )
            # Build JSON regex from schema
            schema_str = JsonSchema(schema).schema
            json_regex = build_regex_from_schema(schema_str)

            # Combine thinking pattern with JSON pattern
            # Allow optional thinking + required JSON
            combined_regex = f"({thinking_pattern})?{json_regex}"

            self.logits_processor = RegexLogitsProcessor(
                combined_regex,
                TransformerTokenizer(tokenizer._tokenizer),
                tensor_library_name="mlx",
            )

            # Store patterns for parsing
            self.thinking_pattern = thinking_pattern
            self.json_regex_pattern = json_regex
        else:
            self.logits_processor = JSONLogitsProcessor(
                schema,
                TransformerTokenizer(tokenizer._tokenizer),
                tensor_library_name="mlx",
            )

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # `tokens` is the full sequence of tokens (prompt + generated).
        # The Outlines processor is stateful and needs the full sequence.

        # Outlines processor expects 1D logits
        logits_1d = logits.reshape(-1)

        # Ensure logits are float32 for the processor
        # The result from outlines is a numpy array, convert it back to mx.array
        processed_logits = self.logits_processor(tokens, logits_1d.astype(mx.float32))

        # Reshape to the original logits shape.
        return mx.array(processed_logits).reshape(logits.shape)
