from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import get_model_path, load, load_config

from ...utils.logger import logger
from ..text_models import BaseTextModel
from .mlx_model import MLXModel
from .model_types import MlxModelCache, ModelId
from .tools.chat_tokenizer import ChatTokenizer
from .tools.hugging_face import HuggingFaceChatTokenizer
from .tools.llama3 import Llama3ChatTokenizer
from .tools.mistral import MistralChatTokenizer

# Initialize global cache objects
_model_cache = None
_mlx_model_cache = None


def load_tools_handler(model_type: str, tokenizer: TokenizerWrapper) -> ChatTokenizer:
    """Factory function to load appropriate tools handler based on model ID."""
    handlers: dict[str, Type[ChatTokenizer]] = {
        # Llama models
        "llama": Llama3ChatTokenizer,
        "mistral": MistralChatTokenizer,
        "qwen2": HuggingFaceChatTokenizer,
    }

    # Get handler class based on model ID or use Llama handler as default
    handler_class = handlers.get(model_type, HuggingFaceChatTokenizer)
    return handler_class(tokenizer)


def load_model(model_id: ModelId) -> BaseTextModel:
    """Load the model and return a BaseTextModel instance.

    Args:
        model_id: ModelId object containing model identification parameters

    Returns:
        Initialized BaseTextModel instance
    """
    global _model_cache, _mlx_model_cache

    # Check if a new model needs to be loaded
    model_needs_reload = _model_cache is None or _model_cache.model_id_obj != model_id

    if model_needs_reload:
        # Cache miss, create a new cache object
        _model_cache = MlxModelCache(model_id)

        # Load configuration and create chat tokenizer
        model_path = get_model_path(model_id.model_id)
        config = load_config(model_path)
        chat_tokenizer = load_tools_handler(
            config["model_type"], _model_cache.tokenizer
        )

        # Create and cache new MLXModel instance
        _mlx_model_cache = MLXModel(model_cache=_model_cache, tokenizer=chat_tokenizer)

    # Return cached model instance
    return _mlx_model_cache
