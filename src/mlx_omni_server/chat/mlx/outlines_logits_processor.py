from typing import List, Union

import mlx.core as mx
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from outlines.models.transformers import TransformerTokenizer
from outlines.processors.structured import JSONLogitsProcessor

from ..schema import ResponseFormat


class OutlinesLogitsProcessor:
    processed_token_count: int = 0

    def __init__(self, tokenizer: TokenizerWrapper, response_format: ResponseFormat):
        json_schema = response_format.json_schema.schema

        # Sanity check the json schema
        # json.loads(json_schema)
        # print(f"json_schema: {json_schema}")

        self.logits_processor = JSONLogitsProcessor(
            json_schema,
            TransformerTokenizer(tokenizer._tokenizer),
        )

    def _convert_to_numpy_int(
        self, tokens: Union[mx.array, List[int], None]
    ) -> np.ndarray:
        """Convert tokens to numpy array of integers."""
        if tokens is None or (
            isinstance(tokens, (list, mx.array)) and len(tokens) == 0
        ):
            return np.array([], dtype=np.int32)

        if isinstance(tokens, list):
            tokens = mx.array(tokens)

        # Convert to numpy array
        if isinstance(tokens, mx.array):
            tokens_np = tokens.tolist()
            return np.array([int(t) for t in tokens_np], dtype=np.int32)

        return np.array([], dtype=np.int32)

    def __call__(self, tokens: mx.array, logits: mx.array):
        # Convert tokens to numpy integers
        generated_tokens = (
            self._convert_to_numpy_int(tokens[-self.processed_token_count :])
            if self.processed_token_count > 0
            else np.array([], dtype=np.int32)
        )

        if logits.dtype == mx.bfloat16:
            logits = logits.astype(mx.float32)
        logits_1d = logits.reshape(-1)

        # Convert logits to numpy array for processing
        logits_np = logits_1d.tolist()
        logits_np = np.array(logits_np)

        # Process with integer tokens
        logits_processed = self.logits_processor(generated_tokens, logits_np)

        # Convert back to mlx array
        logits = mx.array(logits_processed).reshape(1, -1)

        self.processed_token_count += 1
        return logits
