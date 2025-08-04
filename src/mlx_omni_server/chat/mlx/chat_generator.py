"""Chat Generator - Core abstraction layer over mlx-lm for chat completions."""

import time
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from ...utils.logger import logger
from .core_types import (
    CompletionContent,
    CompletionResult,
    GenerationResult,
    GenerationStats,
    StreamContent,
    StreamResult,
)
from .logprobs_processor import LogprobsProcessor
from .model_types import MLXModel

# Default generation parameters
DEFAULT_MAX_TOKENS = 4096


class ChatGenerator:
    """Core chat generator with unified interface for MLX-based text generation.

    This class provides a thin abstraction over mlx-lm's generate functions,
    adding common extensions like tools, reasoning, and caching while keeping
    the interface as close to mlx-lm as possible.
    """

    def __init__(self, model: MLXModel):
        """Initialize with MLX model.

        Args:
            model: MLX model instance containing models and tokenizers
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.chat_template = model.chat_template
        self._prompt_cache = None
        self._logprobs_processor = None

    @classmethod
    def create(
        cls,
        model_id: str,
        adapter_path: Optional[str] = None,
        draft_model_id: Optional[str] = None,
    ) -> "ChatGenerator":
        """Factory method to create ChatGenerator with simplified interface.

        Args:
            model_id: Model name/path (HuggingFace model ID or local path)
            adapter_path: Optional path to LoRA adapter
            draft_model_id: Optional draft model name/path for speculative decoding

        Returns:
            ChatGenerator instance ready for use

        Examples:
            # Simple model loading
            wrapper = ChatGenerator.create("mlx-community/Qwen3-0.6B-4bit")

            # With adapter
            wrapper = ChatGenerator.create(
                model_id="mlx-community/Llama-3.1-8B-Instruct-4bit",
                adapter_path="/path/to/adapter"
            )

            # With draft model for speculative decoding
            wrapper = ChatGenerator.create(
                model_id="mlx-community/Llama-3.1-8B-Instruct-4bit",
                draft_model_id="mlx-community/Qwen3-0.6B-4bit"
            )
        """
        try:
            model = MLXModel.load(
                model_id=model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            )
            return cls(model)
        except Exception as e:
            logger.error(f"Failed to create ChatGenerator: {e}")
            raise

    @classmethod
    def get_or_create(
        cls,
        model_id: str,
        adapter_path: Optional[str] = None,
        draft_model_id: Optional[str] = None,
    ) -> "ChatGenerator":
        """Get or create cached ChatGenerator instance.

        This method provides a unified API for obtaining ChatGenerator instances
        with automatic caching. It will return existing cached instances when
        available, or create new ones as needed.

        Args:
            model_id: Model name/path (HuggingFace model ID or local path)
            adapter_path: Optional path to LoRA adapter
            draft_model_id: Optional draft model name/path for speculative decoding

        Returns:
            Cached or newly created ChatGenerator instance

        Examples:
            # Get or create simple model
            generator = ChatGenerator.get_or_create("mlx-community/Qwen3-0.6B-4bit")

            # With adapter
            generator = ChatGenerator.get_or_create(
                model_id="mlx-community/Llama-3.1-8B-Instruct-4bit",
                adapter_path="/path/to/adapter"
            )

            # With draft model for speculative decoding
            generator = ChatGenerator.get_or_create(
                model_id="mlx-community/Llama-3.1-8B-Instruct-4bit",
                draft_model_id="mlx-community/Qwen3-0.6B-4bit"
            )

        Note:
            This method is thread-safe and uses the global wrapper cache for
            efficient memory usage across different API endpoints.
        """
        # Import here to avoid circular imports
        from .wrapper_cache import wrapper_cache

        return wrapper_cache.get_wrapper(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )

    @property
    def prompt_cache(self):
        """Lazy initialization of prompt cache."""
        if self._prompt_cache is None:
            from .prompt_cache import PromptCache

            self._prompt_cache = PromptCache()
        return self._prompt_cache

    @property
    def logprobs_processor(self):
        """Lazy initialization of logprobs processor."""
        if self._logprobs_processor is None:
            self._logprobs_processor = LogprobsProcessor(self.tokenizer)
        return self._logprobs_processor

    def has_draft_model(self) -> bool:
        """Check if this wrapper has a draft model for speculative decoding."""
        return self.model.has_draft_model()

    def _prepare_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
        json_schema: Optional[Any] = None,
    ) -> str:
        """Prepare prompt using chat tokenizer.

        Args:
            messages: Chat messages in standard format (dictionaries)
            tools: Optional tools for function calling
            template_kwargs: Template parameters for chat tokenizer
            json_schema: JSON schema for structured output (used to detect thinking+schema combination)

        Returns:
            Encoded prompt string
        """
        # Use template_kwargs directly, default to empty dict
        if template_kwargs is None:
            template_kwargs = {}

        # Check if we have thinking + json_schema combination
        # If so, let OutlinesLogitsProcessor handle the <think> tags completely
        enable_thinking = template_kwargs.get("enable_thinking", False)
        if enable_thinking and json_schema is not None:
            template_kwargs["skip_thinking_prefill"] = True

        # Map enable_thinking to enable_thinking_parse for new parameter name
        if "enable_thinking" in template_kwargs:
            template_kwargs["enable_thinking_parse"] = template_kwargs.pop(
                "enable_thinking"
            )

        prompt = self.chat_template.apply_chat_template(
            messages=messages,
            tools=tools,
            **template_kwargs,
        )

        logger.debug(f"Encoded prompt: {prompt}")
        return prompt

    def _create_mlx_kwargs(
        self,
        sampler: Union[Dict[str, Any], Callable, None] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert parameters to mlx-lm compatible kwargs.

        Args:
            sampler: Sampler configuration - can be:
                - Dict: Parameters for make_sampler (temp, top_p, top_k, etc.)
                - Callable: Pre-built sampler function
                - None: Let mlx-lm use its default sampler
            max_tokens: Maximum tokens to generate
            **kwargs: Additional MLX generation parameters

        Returns:
            Dictionary of kwargs for mlx-lm generate functions
        """
        # Core MLX parameters
        mlx_kwargs = {
            "max_tokens": max_tokens,
        }

        # Handle sampler parameter
        if sampler is not None:
            if callable(sampler):
                # Pre-built sampler function
                mlx_kwargs["sampler"] = sampler
            elif isinstance(sampler, dict):
                # Dictionary configuration for make_sampler
                mlx_kwargs["sampler"] = make_sampler(**sampler)
            else:
                raise ValueError(
                    f"Invalid sampler type: {type(sampler)}. Must be dict, callable, or None."
                )
        # If sampler is None, don't set it - let mlx-lm use its defaults

        # Handle special cases that need preprocessing
        # Note: Order matters - processors are applied sequentially
        logits_processors = []

        # 1. Repetition penalty (should come first to avoid repetitive text)
        repetition_penalty = kwargs.pop("repetition_penalty", None)
        if repetition_penalty is not None:
            from mlx_lm.sample_utils import make_logits_processors

            processors = make_logits_processors(repetition_penalty=repetition_penalty)
            logits_processors.extend(processors)

        # 2. JSON schema processor (should come after repetition penalty)
        json_schema = kwargs.pop("json_schema", None)
        if json_schema is not None:
            from .outlines_logits_processor import OutlinesLogitsProcessor

            # Check if we need thinking support
            enable_thinking = self.chat_template.enable_thinking_parse
            logits_processors.append(
                OutlinesLogitsProcessor(
                    self.tokenizer, json_schema, enable_thinking=enable_thinking
                )
            )

        # Add existing processors from kwargs if any
        if "logits_processors" in kwargs:
            existing_processors = kwargs.pop("logits_processors", [])
            if existing_processors:
                logits_processors.extend(existing_processors)

        # Set logits processors if any were created
        if logits_processors:
            mlx_kwargs["logits_processors"] = logits_processors

        # Handle remaining valid kwargs
        for key, value in kwargs.items():
            if value is not None:
                mlx_kwargs[key] = value

        return mlx_kwargs

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        # Core generation parameters
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sampler: Union[Dict[str, Any], Callable, None] = None,
        top_logprobs: Optional[int] = None,
        # Template parameters
        template_kwargs: Optional[Dict[str, Any]] = None,
        # Control parameters
        enable_prompt_cache: bool = False,
        # Additional MLX generation parameters via **kwargs
        **kwargs,
    ) -> CompletionResult:
        """Generate complete response.

        Args:
            messages: Chat messages
            tools: Optional tools for function calling
            max_tokens: Maximum tokens to generate
            sampler: Sampler configuration - can be:
                - Dict: Parameters for make_sampler (temp, top_p, top_k, etc.)
                - Callable: Pre-built sampler function
                - None: Let mlx-lm use its default sampler
            top_logprobs: Number of top logprobs to include (None to disable)
            template_kwargs: Template parameters for chat tokenizer (enable_thinking, thinking_budget, etc.)
            enable_prompt_cache: Enable prompt caching
            **kwargs: Additional MLX generation parameters (max_kv_size, kv_bits, repetition_penalty, etc.)

        Returns:
            Complete generation result
        """
        try:
            # Generate complete response by collecting stream.
            # stream_generate handles the preparation of configurations.
            complete_raw_text = ""
            final_stream_result = None
            all_text_tokens = []
            all_reasoning_tokens = []

            for stream_result in self.generate_stream(
                messages,
                tools,
                max_tokens,
                sampler,
                top_logprobs,
                template_kwargs,
                enable_prompt_cache,
                **kwargs,
            ):
                # Collect deltas to reconstruct complete content
                if stream_result.content.text_delta:
                    complete_raw_text += stream_result.content.text_delta
                    all_text_tokens.append(stream_result.content.token)
                elif stream_result.content.reasoning_delta:
                    # Reasoning tokens should also be included in complete_raw_text
                    complete_raw_text += stream_result.content.reasoning_delta
                    all_reasoning_tokens.append(stream_result.content.token)

                final_stream_result = stream_result

            if final_stream_result is None:
                raise RuntimeError("No tokens generated")

            logger.info(f"Model Response:\n{complete_raw_text}")
            chat_result = self.chat_template.parse_chat_response(complete_raw_text)

            # Determine appropriate finish_reason
            finish_reason = final_stream_result.finish_reason
            if chat_result.tool_calls:
                finish_reason = "tools"

            # Create CompletionContent with complete data
            content = CompletionContent(
                text=chat_result.content,
                reasoning=chat_result.thinking,
                tool_calls=chat_result.tool_calls,
                text_tokens=all_text_tokens,
                reasoning_tokens=all_reasoning_tokens if all_reasoning_tokens else None,
            )

            # Return final result with all processing applied
            return GenerationResult(
                content=content,
                finish_reason=finish_reason,
                stats=final_stream_result.stats,
                logprobs=final_stream_result.logprobs,
                from_draft=final_stream_result.from_draft,
            )

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        # Core generation parameters
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sampler: Union[Dict[str, Any], Callable, None] = None,
        top_logprobs: Optional[int] = None,
        # Template parameters
        template_kwargs: Optional[Dict[str, Any]] = None,
        # Control parameters
        enable_prompt_cache: bool = False,
        # Additional MLX generation parameters via **kwargs
        **kwargs,
    ) -> Generator[StreamResult, None, None]:
        """Generate streaming response.

        Args:
            messages: Chat messages
            tools: Optional tools for function calling
            max_tokens: Maximum tokens to generate
            sampler: Sampler configuration - can be:
                - Dict: Parameters for make_sampler (temp, top_p, top_k, etc.)
                - Callable: Pre-built sampler function
                - None: Let mlx-lm use its default sampler
            top_logprobs: Number of top logprobs to include (None to disable)
            template_kwargs: Template parameters for chat tokenizer (enable_thinking, thinking_budget, etc.)
            enable_prompt_cache: Enable prompt caching
            **kwargs: Additional MLX generation parameters (max_kv_size, kv_bits, repetition_penalty, etc.)

        Yields:
            Streaming generation results
        """
        # Record start time for first token latency measurement
        request_start_time = time.perf_counter()
        first_token_time = None

        try:

            # Extract json_schema from kwargs for coordination with chat_template
            json_schema = kwargs.get("json_schema")

            # Prepare prompt
            prompt = self._prepare_prompt(messages, tools, template_kwargs, json_schema)

            # Tokenize prompt
            tokenized_prompt = self.tokenizer.encode(prompt)

            # Process cache if enabled
            processed_prompt = tokenized_prompt
            cached_tokens = 0

            if enable_prompt_cache:
                processed_prompt, cached_tokens = self.prompt_cache.get_prompt_cache(
                    self.model, tokenized_prompt
                )

            # Create MLX kwargs
            mlx_kwargs = self._create_mlx_kwargs(
                sampler=sampler,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Add cache to kwargs if available
            if enable_prompt_cache and self.prompt_cache.cache:
                mlx_kwargs["prompt_cache"] = self.prompt_cache.cache

            # Stream generation
            generated_tokens = []

            for response in stream_generate(
                model=self.model.model,
                tokenizer=self.tokenizer,
                prompt=processed_prompt,
                draft_model=self.model.draft_model,
                **mlx_kwargs,
            ):
                if response.finish_reason is not None:
                    break

                generated_tokens.append(response.token)

                # Record first token time if this is the first token
                if first_token_time is None:
                    first_token_time = time.perf_counter() - request_start_time

                # Process logprobs if requested
                logprobs = None
                if top_logprobs is not None:
                    logprobs = self.logprobs_processor.get_logprobs(
                        response, top_logprobs
                    )

                parse_result = self.chat_template.stream_parse_chat_result(
                    response.text
                )

                # Create StreamContent based on parse result
                chunk_index = len(generated_tokens)

                # Determine which delta field to populate
                if parse_result.thinking:
                    content = StreamContent(
                        reasoning_delta=parse_result.thinking,
                        token=response.token,
                        chunk_index=chunk_index,
                    )
                else:
                    content = StreamContent(
                        text_delta=parse_result.content or response.text,
                        token=response.token,
                        chunk_index=chunk_index,
                    )

                stats = GenerationStats(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.generation_tokens,
                    prompt_tps=response.prompt_tps,
                    generation_tps=response.generation_tps,
                    peak_memory=response.peak_memory,
                    cache_hit_tokens=cached_tokens,
                    time_to_first_token=first_token_time or 0.0,
                )

                yield GenerationResult(
                    content=content,
                    finish_reason=response.finish_reason,
                    stats=stats,
                    logprobs=logprobs,
                    from_draft=response.from_draft,
                )

            # Extend cache with generated tokens if caching is enabled
            if enable_prompt_cache and generated_tokens:
                self.prompt_cache.extend_completion_cache(generated_tokens)

        except Exception as e:
            logger.error(f"Error during stream generation: {e}")
            raise RuntimeError(f"Stream generation failed: {e}")
