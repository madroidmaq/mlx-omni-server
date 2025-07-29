# Initialize global cache objects
# _cached_model: MLXModel = None
# _cached_adapter: OpenAIAdapter = None
# _cached_anthropic_adapter: AnthropicMessagesAdapter = None

#
# def load_openai_adapter(model_key: MLXModel) -> OpenAIAdapter:
#     """Load the model and return an OpenAIAdapter instance.
#
#     Args:
#         model_key: MLXModel object containing model identification parameters
#
#     Returns:
#         Initialized OpenAIAdapter instance
#     """
#     global _cached_model, _cached_adapter
#
#     # Check if a new model needs to be loaded
#     model_needs_reload = _cached_model is None or _cached_model != model_key
#
#     if model_needs_reload:
#         # Cache miss, use the already loaded model
#         _cached_model = model_key
#
#         # Create and cache new OpenAIAdapter instance
#         _cached_adapter = OpenAIAdapter(model=_cached_model)
#
#     # Return cached adapter instance
#     return _cached_adapter


# def load_anthropic_adapter(model_key: MLXModel) -> AnthropicMessagesAdapter:
#     """Load the model and return an AnthropicMessagesAdapter instance.
#
#     Args:
#         model_key: MLXModel object containing model identification parameters
#
#     Returns:
#         Initialized AnthropicMessagesAdapter instance
#     """
#     global _cached_model, _cached_anthropic_adapter
#
#     # Check if a new model needs to be loaded
#     model_needs_reload = _cached_model is None or _cached_model != model_key
#
#     if model_needs_reload:
#         # Cache miss, use the already loaded model
#         _cached_model = model_key
#
#         # Create and cache new AnthropicMessagesAdapter instance
#         _cached_anthropic_adapter = AnthropicMessagesAdapter(model=_cached_model)
#     elif _cached_anthropic_adapter is None:
#         # Model is cached but anthropic adapter is not
#         _cached_anthropic_adapter = AnthropicMessagesAdapter(model=_cached_model)
#
#     # Return cached adapter instance
#     return _cached_anthropic_adapter
