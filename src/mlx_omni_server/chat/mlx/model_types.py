from dataclasses import dataclass
from typing import Optional

from mlx_lm.utils import get_model_path, load, load_config

from ...utils.logger import logger


@dataclass(frozen=True)
class ModelId:
    """Entity class representing a unique model identifier.

    This class encapsulates all parameters that determine whether a model
    needs to be reloaded. It replaces the tuple-based approach with a more
    object-oriented design that's easier to extend and maintain.
    """

    model_id: str
    adapter_path: Optional[str] = None
    draft_model: Optional[str] = None

    def __str__(self) -> str:
        """Return a string representation of the model ID for debugging."""
        parts = [f"model_id={self.model_id}"]
        if self.adapter_path:
            parts.append(f"adapter_path={self.adapter_path}")
        if self.draft_model:
            parts.append(f"draft_model={self.draft_model}")
        return f"ModelId({', '.join(parts)})"


class MlxModelCache:
    """Model cache class to avoid reloading the same models.

    This class manages the cache of main models and draft models based on ModelId objects.
    """

    def __init__(self, model_id: ModelId = None):
        """Initialize the cache object.

        Args:
            model_id: Optional ModelId object for initialization
        """
        self.model_id_obj = model_id
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.draft_tokenizer = None

        # If a model ID is provided, load the models directly
        if model_id:
            self._load_models()

    def _load_models(self):
        """Load the main model and draft model (if needed)."""
        # Load the main model
        self.model, self.tokenizer = load(
            self.model_id_obj.model_id,
            tokenizer_config={"trust_remote_code": True},
            adapter_path=self.model_id_obj.adapter_path,
        )
        logger.info(f"Loaded new model: {self.model_id_obj.model_id}")

        # If needed, load the draft model
        if self.model_id_obj.draft_model:
            self.draft_model, self.draft_tokenizer = load(
                self.model_id_obj.draft_model,
                tokenizer_config={"trust_remote_code": True},
            )

            # Check if vocabulary sizes match
            if self.draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
                logger.warn(
                    f"Draft model({self.model_id_obj.draft_model}) tokenizer does not match model tokenizer."
                )

            logger.info(f"Loaded new draft model: {self.model_id_obj.draft_model}")
