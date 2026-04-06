"""MLX Model types and management."""

from pathlib import Path
from typing import Optional, Union

import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load, load_config

# Handle mlx_lm version compatibility: newer versions use hf_repo_to_path (returns Path),
# older versions (< 0.29) use get_model_path (returns Tuple[Path, Optional[str]])
#
# Additionally: our deployments often pass a local filesystem path as model_id.
# Newer huggingface_hub validators reject absolute paths as "repo ids"; treat real
# local paths as paths and bypass hf_repo_to_path/get_model_path.
try:
    from mlx_lm.utils import hf_repo_to_path as _get_model_path

    def get_model_path(model_id: str) -> Path:
        """Get model path (wrapper for newer mlx_lm versions)."""
        p = Path(model_id)
        if p.is_absolute() and p.exists():
            return p
        return _get_model_path(model_id)

except ImportError:
    from mlx_lm.utils import get_model_path as _get_model_path

    def get_model_path(model_id: str) -> Path:
        """Get model path (wrapper for older mlx_lm versions)."""
        p = Path(model_id)
        if p.is_absolute() and p.exists():
            return p
        result = _get_model_path(model_id)
        # Old version returns Tuple[Path, Optional[str]], extract just the Path
        if isinstance(result, tuple):
            return result[0]
        elif isinstance(result, Path):
            return result
        else:
            raise TypeError(f"Unexpected return type from get_model_path: {type(result)}")


# Models that require mlx_vlm instead of mlx_lm
MLX_VLM_ONLY_MODELS = {"gemma4"}


def _is_vlm_model(model_id: str, config: dict) -> bool:
    """Check if model should use mlx_vlm."""
    model_type = config.get("model_type", "")
    
    # Check explicit VLM model list
    if model_type in MLX_VLM_ONLY_MODELS:
        return True
    
    # Check model ID patterns for known VLM models
    model_id_lower = model_id.lower()
    for pattern in MLX_VLM_ONLY_MODELS:
        if pattern in model_id_lower:
            return True
    
    return False


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
    from ...utils.logger import logger
    from .tools.chat_template import ChatTemplate

    if not model_id or not model_id.strip():
        raise ValueError("model_id cannot be empty")

    model_id = model_id.strip()

    try:
        # Load the main model
        model_path = get_model_path(model_id)
        config = load_config(model_path)

        # Check if we need to use mlx_vlm for this model
        if _is_vlm_model(model_id, config):
            # Use mlx_vlm for Gemma 4 and other VLM-only models
            logger.info(f"Loading {model_id} using mlx_vlm (model_type: {config.get('model_type')})")
            
            # Warn if draft model is requested - not supported for VLM models
            if draft_model_id:
                logger.warning(
                    f"Speculative decoding (draft_model_id={draft_model_id}) is not supported "
                    f"for VLM-only models like {model_id}. Proceeding without draft model."
                )
            
            try:
                from mlx_vlm import load as vlm_load
                
                # Load using mlx_vlm
                vlm_model, vlm_processor = vlm_load(model_id)
                
                # Create a wrapper for the VLM model to match expected interface
                class VLMModelWrapper:
                    """Wrapper to make mlx_vlm model compatible with mlx-lm interface."""
                    
                    def __init__(self, model, processor):
                        self.model = model
                        self.processor = processor
                        self.config = model.config if hasattr(model, 'config') else {}
                    
                    def __call__(self, *args, **kwargs):
                        return self.model(*args, **kwargs)
                
                class VLMTokenizerWrapper:
                    """Wrapper to make mlx_vlm processor compatible with tokenizer interface."""
                    
                    def __init__(self, processor):
                        self.processor = processor
                        # Try to get the underlying tokenizer
                        if hasattr(processor, 'tokenizer'):
                            self.tokenizer = processor.tokenizer
                        else:
                            self.tokenizer = processor
                        
                        # Load config from model path for apply_chat_template
                        # We'll get it from the model's config attribute if available
                        self.config = {"model_type": "gemma4"}  # Default
                        if hasattr(processor, 'model') and hasattr(processor.model, 'config'):
                            model_config = processor.model.config
                            if hasattr(model_config, '__dict__'):
                                self.config = model_config.__dict__
                            elif isinstance(model_config, dict):
                                self.config = model_config
                    
                    def apply_chat_template(self, *args, **kwargs):
                        # Use mlx_vlm's apply_chat_template
                        from mlx_vlm.prompt_utils import apply_chat_template as mlx_vlm_template
                        
                        # Get messages/conversation from args or kwargs
                        messages = kwargs.pop('messages', kwargs.pop('conversation', None))
                        if messages is None and args:
                            messages = args[0]
                            args = args[1:]
                        
                        # Build proper message list for mlx_vlm
                        formatted_messages = []
                        if isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict):
                                    formatted_messages.append(msg)
                                elif isinstance(msg, str):
                                    formatted_messages.append({"role": "user", "content": msg})
                        
                        # Get config from stored config
                        config = self.config
                        
                        # Don't force add_generation_prompt - let caller control it
                        add_generation_prompt = kwargs.pop("add_generation_prompt", None)
                        template_kwargs = dict(kwargs)
                        if add_generation_prompt is not None:
                            template_kwargs["add_generation_prompt"] = add_generation_prompt
                        
                        return mlx_vlm_template(
                            self.processor,
                            config,
                            formatted_messages,
                            **template_kwargs
                        )
                    
                    @property
                    def vocab_size(self):
                        return len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 0
                    
                    def __call__(self, *args, **kwargs):
                        return self.tokenizer(*args, **kwargs)
                    
                    def encode(self, *args, **kwargs):
                        return self.tokenizer.encode(*args, **kwargs) if hasattr(self.tokenizer, 'encode') else []
                    
                    def decode(self, *args, **kwargs):
                        return self.tokenizer.decode(*args, **kwargs) if hasattr(self.tokenizer, 'decode') else ""
                
                # Wrap the model and processor
                wrapped_model = VLMModelWrapper(vlm_model, vlm_processor)
                wrapped_tokenizer = VLMTokenizerWrapper(vlm_processor)
                
                # Create chat template
                # Get model_type from processor's model config
                model_type = "gemma4"  # Default
                if hasattr(vlm_processor, 'model') and hasattr(vlm_processor.model, 'config'):
                    model_config = vlm_processor.model.config
                    if hasattr(model_config, 'model_type'):
                        model_type = model_config.model_type
                    elif hasattr(model_config, '__dict__') and 'model_type' in model_config.__dict__:
                        model_type = model_config.__dict__['model_type']
                chat_template = ChatTemplate(model_type, wrapped_tokenizer)
                
                logger.info(f"Loaded {model_id} with mlx_vlm successfully")
                
                return MLXModel(
                    model_id=model_id,
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                    model=wrapped_model,
                    tokenizer=wrapped_tokenizer,
                    chat_template=chat_template,
                    is_vlm_model=True,
                )
                
            except ImportError as e:
                logger.error(f"mlx_vlm not available: {e}")
                raise RuntimeError(f"Model {model_id} requires mlx_vlm but it failed to import: {e}")

        # Standard mlx_lm loading for other models
        model, tokenizer = load(
            model_id,
            tokenizer_config={"trust_remote_code": True},
            adapter_path=adapter_path,
        )
        logger.info(f"Loaded model: {model_id}")

        # Load configuration and create chat tokenizer
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
                    logger.warning(
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
        chat_template: "ChatTemplate",
        draft_model: Optional[nn.Module] = None,
        draft_tokenizer: Optional[TokenizerWrapper] = None,
        is_vlm_model: bool = False,
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
            is_vlm_model: Whether this is a VLM-only model (like Gemma 4)
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
        self.is_vlm_model = is_vlm_model

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
        if self.is_vlm_model:
            parts.append("is_vlm_model=True")
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
