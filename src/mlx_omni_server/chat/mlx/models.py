from mlx_omni_server.chat.openai_adapter import OpenAIAdapter

from .model_types import MlxModelCache, ModelId

# Initialize global cache objects
_model_cache = None
_mlx_model_cache = None


def load_openai_adapter(model_id: ModelId) -> OpenAIAdapter:
    """Load the model and return a BaseTextModel instance.

    Args:
        model_id: ModelId object containing model identification parameters

    Returns:
        Initialized BaseTextModel instance
    """
    global _model_cache, _mlx_model_cache

    # Check if a new model needs to be loaded
    model_needs_reload = _model_cache is None or _model_cache.model_id != model_id

    if model_needs_reload:
        # Cache miss, create a new cache object
        _model_cache = MlxModelCache(model_id)

        # Create and cache new MLXModel instance
        _mlx_model_cache = OpenAIAdapter(model_cache=_model_cache)

    # Return cached model instance
    return _mlx_model_cache
