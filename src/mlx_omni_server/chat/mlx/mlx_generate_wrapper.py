"""MLX Generate Wrapper - Core abstraction layer over mlx-lm."""

from typing import Any, Dict, Generator, List, Optional

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from ...utils.logger import logger
from .core_types import GenerationResult, GenerationStats
from .logprobs_processor import LogprobsProcessor
from .model_types import MlxModelCache


class MLXGenerateWrapper:
    """Core wrapper around mlx-lm with unified interface.

    This class provides a thin abstraction over mlx-lm's generate functions,
    adding common extensions like tools, reasoning, and caching while keeping
    the interface as close to mlx-lm as possible.
    """

    def __init__(self, model_cache: MlxModelCache):
        """Initialize with model cache.

        Args:
            model_cache: MLX model cache containing models and tokenizers
        """
        self.model_cache = model_cache
        self.tokenizer = model_cache.tokenizer
        self.chat_template = model_cache.chat_template
        self._prompt_cache = None
        self._logprobs_processor = None

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

    def _prepare_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Prepare prompt using chat tokenizer.

        Args:
            messages: Chat messages in standard format (dictionaries)
            tools: Optional tools for function calling
            template_kwargs: Template parameters for chat tokenizer

        Returns:
            Encoded prompt string
        """
        if tools:
            logger.debug(f"Prepared {len(tools)} tools for encoding")

        # Use template_kwargs directly, default to empty dict
        if template_kwargs is None:
            template_kwargs = {}

        prompt = self.chat_template.apply_chat_template(
            messages=messages,
            tools=tools,
            **template_kwargs,
        )

        logger.debug(f"Encoded prompt: {prompt}")
        return prompt

    def _create_mlx_kwargs(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        sampler_kwargs: Optional[Dict[str, Any]],
        max_tokens: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert parameters to mlx-lm compatible kwargs.

        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling value
            top_k: Top-k sampling value
            sampler_kwargs: Additional sampler parameters
            max_tokens: int = 2048,
            **kwargs

        Returns:
            Dictionary of kwargs for mlx-lm generate functions
        """
        # Core MLX parameters
        mlx_kwargs = {
            "max_tokens": max_tokens,
        }

        # Create sampler with core parameters
        core_sampler_kwargs = {
            "temp": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        # Merge with additional sampler kwargs
        if sampler_kwargs:
            core_sampler_kwargs.update(sampler_kwargs)

        mlx_kwargs["sampler"] = make_sampler(**core_sampler_kwargs)

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

            logits_processors.append(
                OutlinesLogitsProcessor(self.tokenizer, json_schema)
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
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        top_logprobs: Optional[int] = None,
        # Additional sampler parameters
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        # Template parameters
        template_kwargs: Optional[Dict[str, Any]] = None,
        # Control parameters
        enable_prompt_cache: bool = False,
        # Additional MLX generation parameters via **kwargs
        **kwargs,
    ) -> GenerationResult:
        """Generate complete response.

        Args:
            messages: Chat messages
            tools: Optional tools for function calling
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling value
            top_k: Top-k sampling value
            top_logprobs: Number of top logprobs to include (None to disable)
            sampler_kwargs: Additional sampler parameters for make_sampler
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
            final_result_from_stream = None

            for result in self.stream_generate(
                messages,
                tools,
                max_tokens,
                temperature,
                top_p,
                top_k,
                top_logprobs,
                sampler_kwargs,
                template_kwargs,
                enable_prompt_cache,
                **kwargs,
            ):
                if result.raw_delta:
                    complete_raw_text += result.raw_delta
                final_result_from_stream = result

            if final_result_from_stream is None:
                raise RuntimeError("No tokens generated")

            logger.info(f"Model Response:\n{complete_raw_text}")
            chat_result = self.chat_template.parse_chat_response(complete_raw_text)

            # Determine appropriate finish_reason
            finish_reason = final_result_from_stream.finish_reason
            if chat_result.tool_calls:
                finish_reason = "tools"

            # Return final result with all processing applied
            return GenerationResult(
                text=chat_result.content,
                token=final_result_from_stream.token,
                finish_reason=finish_reason,
                stats=final_result_from_stream.stats,
                tool_calls=chat_result.tool_calls,
                reasoning=chat_result.thinking,
                logprobs=final_result_from_stream.logprobs,
                from_draft=final_result_from_stream.from_draft,
            )

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        # Core generation parameters
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        top_logprobs: Optional[int] = None,
        # Additional sampler parameters
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        # Template parameters
        template_kwargs: Optional[Dict[str, Any]] = None,
        # Control parameters
        enable_prompt_cache: bool = False,
        # Additional MLX generation parameters via **kwargs
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate streaming response.

        Args:
            messages: Chat messages
            tools: Optional tools for function calling
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling value
            top_k: Top-k sampling value
            top_logprobs: Number of top logprobs to include (None to disable)
            sampler_kwargs: Additional sampler parameters for make_sampler
            template_kwargs: Template parameters for chat tokenizer (enable_thinking, thinking_budget, etc.)
            enable_prompt_cache: Enable prompt caching
            **kwargs: Additional MLX generation parameters (max_kv_size, kv_bits, repetition_penalty, etc.)

        Yields:
            Streaming generation results
        """
        try:

            # Prepare prompt
            prompt = self._prepare_prompt(messages, tools, template_kwargs)

            # Tokenize prompt
            tokenized_prompt = self.tokenizer.encode(prompt)

            # Process cache if enabled
            processed_prompt = tokenized_prompt
            cached_tokens = 0

            if enable_prompt_cache:
                processed_prompt, cached_tokens = self.prompt_cache.get_prompt_cache(
                    self.model_cache, tokenized_prompt
                )

            # Create MLX kwargs
            mlx_kwargs = self._create_mlx_kwargs(
                temperature,
                top_p,
                top_k,
                sampler_kwargs,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Add cache to kwargs if available
            if enable_prompt_cache and self.prompt_cache.cache:
                mlx_kwargs["prompt_cache"] = self.prompt_cache.cache

            # Stream generation
            generated_tokens = []

            for response in stream_generate(
                model=self.model_cache.model,
                tokenizer=self.tokenizer,
                prompt=processed_prompt,
                draft_model=self.model_cache.draft_model,
                **mlx_kwargs,
            ):
                if response.finish_reason is not None:
                    break

                generated_tokens.append(response.token)

                # Process logprobs if requested
                logprobs = None
                if top_logprobs is not None:
                    logprobs = self.logprobs_processor.get_logprobs(
                        response, top_logprobs
                    )

                parse_result = self.chat_template.stream_parse_chat_result(
                    response.text
                )

                stats = GenerationStats(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.generation_tokens,
                    prompt_tps=response.prompt_tps,
                    generation_tps=response.generation_tps,
                    peak_memory=response.peak_memory,
                    cache_hit_tokens=cached_tokens,
                )
                yield GenerationResult(
                    text=parse_result.content,
                    token=response.token,
                    finish_reason=response.finish_reason,
                    stats=stats,
                    tool_calls=None,
                    reasoning=parse_result.thinking,
                    logprobs=logprobs,
                    from_draft=response.from_draft,
                    raw_delta=response.text,
                )

            # Extend cache with generated tokens if caching is enabled
            if enable_prompt_cache and generated_tokens:
                self.prompt_cache.extend_completion_cache(generated_tokens)

        except Exception as e:
            logger.error(f"Error during stream generation: {e}")
            raise RuntimeError(f"Stream generation failed: {e}")
