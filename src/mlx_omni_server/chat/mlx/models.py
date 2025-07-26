from mlx_omni_server.chat.openai_adapter import OpenAIAdapter

from .model_types import MLXModel

# Initialize global cache objects
_cached_model: MLXModel = None
_cached_adapter: OpenAIAdapter = None


def load_openai_adapter(model_key: MLXModel) -> OpenAIAdapter:
    """Load the model and return an OpenAIAdapter instance.

    Args:
        model_key: MLXModel object containing model identification parameters

    Returns:
        Initialized OpenAIAdapter instance
    """
    global _cached_model, _cached_adapter

    # Check if a new model needs to be loaded
    model_needs_reload = _cached_model is None or _cached_model != model_key

    if model_needs_reload:
        # Cache miss, use the already loaded model
        _cached_model = model_key

        # Create and cache new OpenAIAdapter instance
        _cached_adapter = OpenAIAdapter(model=_cached_model)

    # Return cached adapter instance
    return _cached_adapter
