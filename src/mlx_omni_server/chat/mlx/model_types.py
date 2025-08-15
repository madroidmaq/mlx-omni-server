"""MLX Model types and management."""

from typing import Optional

import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import get_model_path, load, load_config

from ...utils.logger import logger
from .tools.chat_template import ChatTemplate


def load_mlx_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model_id: Optional[str] = None,
) -> "MLXModel":
    """Factory function to load MLX models.

    Args:
        model_id: Model name/path (HuggingFace model ID or local path)
        adapter_path: Optional path to LoRA adapter
        draft_model_id: Optional draft model name/path for speculative decoding

    Returns:
        MLXModel instance with loaded models

    Raises:
        ValueError: If model_id is invalid
        RuntimeError: If model loading fails
    """
    if not model_id or not model_id.strip():
        raise ValueError("model_id cannot be empty")

    model_id = model_id.strip()

    try:
        # Load the main model
        model, tokenizer = load(
            model_id,
            tokenizer_config={"trust_remote_code": True},
            adapter_path=adapter_path,
        )
        logger.info(f"Loaded model: {model_id}")

        # Load configuration and create chat tokenizer
        model_path = get_model_path(model_id)[0]
        config = load_config(model_path)
        chat_template = ChatTemplate(config["model_type"], tokenizer)

        # Load draft model if specified
        draft_model = None
        draft_tokenizer = None
        if draft_model_id:
            try:
                draft_model, draft_tokenizer = load(
                    draft_model_id,
                    tokenizer_config={"trust_remote_code": True},
                )

                # Check if vocabulary sizes match
                if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                    logger.warn(
                        f"Draft model({draft_model_id}) tokenizer does not match model tokenizer."
                    )

                logger.info(f"Loaded draft model: {draft_model_id}")
            except Exception as e:
                logger.error(f"Failed to load draft model {draft_model_id}: {e}")
                # Continue without draft model
                draft_model = None
                draft_tokenizer = None

        return MLXModel(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            model=model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
        )

    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise RuntimeError(f"Model loading failed for {model_id}: {e}") from e


class MLXModel:
    """Simplified MLX model container.

    This class is a simple data container for loaded MLX models.
    For model management operations, create new instances rather than modifying existing ones.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str],
        draft_model_id: Optional[str],
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        chat_template: ChatTemplate,
        draft_model: Optional[nn.Module] = None,
        draft_tokenizer: Optional[TokenizerWrapper] = None,
    ):
        """Initialize MLX model container.

        This constructor is typically called by load_mlx_model() factory function.

        Args:
            model_id: Model name/path
            adapter_path: Path to LoRA adapter (if any)
            draft_model_id: Draft model name/path (if any)
            model: Loaded main model
            tokenizer: Loaded tokenizer
            chat_template: Chat template instance
            draft_model: Loaded draft model (optional)
            draft_tokenizer: Draft model tokenizer (optional)
        """
        # Model identification
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.draft_model_id = draft_model_id

        # Loaded model components
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

    @classmethod
    def load(
        cls,
        model_id: str,
        adapter_path: Optional[str] = None,
        draft_model_id: Optional[str] = None,
    ) -> "MLXModel":
        return load_mlx_model(model_id, adapter_path, draft_model_id)

    def __str__(self) -> str:
        """Return a string representation of the model for debugging."""
        parts = [f"model_id={self.model_id}"]
        if self.adapter_path:
            parts.append(f"adapter_path={self.adapter_path}")
        if self.draft_model_id:
            parts.append(f"draft_model_id={self.draft_model_id}")
        return f"MLXModel({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        """Check equality based on model configuration."""
        if not isinstance(other, MLXModel):
            return False
        return (
            self.model_id == other.model_id
            and self.adapter_path == other.adapter_path
            and self.draft_model_id == other.draft_model_id
        )

    def __hash__(self) -> int:
        """Hash based on model configuration for use as dict keys."""
        return hash((self.model_id, self.adapter_path, self.draft_model_id))

    def has_adapter(self) -> bool:
        """Check if this model has an adapter configured."""
        return self.adapter_path is not None

    def has_draft_model(self) -> bool:
        """Check if draft model is available."""
        return self.draft_model is not None and self.draft_tokenizer is not None
