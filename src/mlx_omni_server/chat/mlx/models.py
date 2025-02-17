from typing import Optional, Type

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import get_model_path, load, load_config

from ...utils.logger import logger
from ..text_models import BaseTextModel
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


def load_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model_id: Optional[str] = None,
) -> BaseTextModel:
    """Load a model and tokenizer from the given model ID."""
    model, tokenizer = load(
        model_id,
        tokenizer_config={"trust_remote_code": True},
        adapter_path=adapter_path,
    )

    if draft_model_id:
        draft_model, draft_tokenizer = load(
            draft_model_id,
            tokenizer_config={"trust_remote_code": True},
        )
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            logger.warn(
                f"Draft model({draft_model}) tokenizer does not match model tokenizer."
            )

    model_path = get_model_path(model_id)
    config = load_config(model_path)

    chat_tokenizer = load_tools_handler(config["model_type"], tokenizer)

    if draft_model_id:
        logger.info(
            f"Initialized MLXModel with model_id: {model_id}, and with draft model: {draft_model_id}"
        )
    else:
        logger.info(f"Initialized MLXModel with model_id: {model_id}")

    return MLXModel(
        model_id=model_id,
        model=model,
        tokenizer=chat_tokenizer,
        draft_model=None if draft_model_id is None else draft_model,
    )
