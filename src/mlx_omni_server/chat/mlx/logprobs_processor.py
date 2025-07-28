"""Log probabilities processing utilities for MLX responses."""

from typing import Any, Dict, Optional

import mlx.core as mx


class LogprobsProcessor:
    """Handles logprobs processing for MLX generation responses."""

    def __init__(self, tokenizer):
        """Initialize with tokenizer for token decoding.

        Args:
            tokenizer: MLX tokenizer for decoding tokens
        """
        self.tokenizer = tokenizer

    def process_logprobs(
        self, response, top_k: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Process logprobs from MLX response.

        Args:
            response: MLX response object
            top_k: Number of top logprobs to include

        Returns:
            Processed logprobs dictionary or None
        """
        if not hasattr(response, "logprobs") or response.logprobs is None:
            return None

        current_token = response.token
        current_logprobs = response.logprobs

        token_str = self.tokenizer.decode([current_token])
        token_logprob = mx.clip(
            current_logprobs[current_token], a_min=-100, a_max=None
        ).item()
        token_bytes = token_str.encode("utf-8")

        token_info = {
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_bytes),
        }

        top_logprobs = []
        if top_k is not None:
            top_indices = mx.argpartition(-current_logprobs, kth=top_k - 1)[:top_k]
            top_probs = mx.clip(current_logprobs[top_indices], a_min=-100, a_max=None)

            for idx, logprob in zip(top_indices.tolist(), top_probs.tolist()):
                token = self.tokenizer.decode([idx])
                token_bytes = token.encode("utf-8")
                top_logprobs.append(
                    {"token": token, "logprob": logprob, "bytes": list(token_bytes)}
                )

        return {**token_info, "top_logprobs": top_logprobs}

    def get_logprobs(
        self, response, top_logprobs: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Get logprobs from response if requested.

        Args:
            response: MLX response object
            top_logprobs: Number of top logprobs to include

        Returns:
            Processed logprobs dictionary or None
        """
        if top_logprobs is not None:
            return self.process_logprobs(response, top_logprobs)
        return None
