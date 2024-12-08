from typing import Type

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import get_model_path, load, load_config

from .base_models import BaseMLXModel
from .mlx_model import MLXModel
from .tools.chat_tokenizer import ChatTokenizer
from .tools.hugging_face import HuggingFaceChatTokenizer
from .tools.llama3 import Llama3ChatTokenizer
from .tools.mistral import MistralChatTokenizer


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


def load_model(model_id: str) -> BaseMLXModel:
    """Load a model and tokenizer from the given model ID."""
    model, tokenizer = load(
        model_id,
        tokenizer_config={"trust_remote_code": True},
    )

    model_path = get_model_path(model_id)
    config = load_config(model_path)

    chat_tokenizer = load_tools_handler(config["model_type"], tokenizer)

    return MLXModel(model_id=model_id, model=model, tokenizer=chat_tokenizer)
