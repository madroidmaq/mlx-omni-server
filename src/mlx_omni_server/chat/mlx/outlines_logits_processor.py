import json
from typing import Any, Dict, Optional, Type, Union

import mlx.core as mx
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from outlines.models.transformers import TransformerTokenizer
from outlines.processors.structured import JSONLogitsProcessor
from pydantic import BaseModel


class OutlinesLogitsProcessor:
    processed_token_count: int = 0

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        schema: Union[Dict[str, Any], Type[BaseModel], str],
    ):
        """Initialize the OutlinesLogitsProcessor.

        Args:
            tokenizer: MLX tokenizer wrapper
            schema: JSON schema dictionary, Pydantic BaseModel class, or JSON schema string
            whitespace_pattern: Pattern to use for JSON syntactic whitespace
        """
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
