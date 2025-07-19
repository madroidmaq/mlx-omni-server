"""MLX Generate Wrapper - Core abstraction layer over mlx-lm."""

from dataclasses import replace
from typing import Any, Dict, Generator, List, Optional

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from ...utils.logger import logger
from .core_types import GenerationResult, GenerationStats
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
        self.chat_tokenizer = model_cache.chat_tokenizer

        # Processors will be initialized when needed
        self._reasoning_decoder = None
        self._prompt_cache = None

    @property
    def reasoning_decoder(self):
        """Lazy initialization of reasoning decoder."""
        if self._reasoning_decoder is None:
            from .tools.reasoning_decoder import ReasoningDecoder

            self._reasoning_decoder = ReasoningDecoder(self.tokenizer)
        return self._reasoning_decoder

    @property
    def prompt_cache(self):
        """Lazy initialization of prompt cache."""
        if self._prompt_cache is None:
            from .prompt_cache import PromptCache

            self._prompt_cache = PromptCache()
        return self._prompt_cache

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

        prompt = self.chat_tokenizer.encode(
            messages=messages, tools=tools, **template_kwargs
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
        top_logprobs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert parameters to mlx-lm compatible kwargs.

        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling value
            top_k: Top-k sampling value
            sampler_kwargs: Additional sampler parameters
            max_tokens: int = 2048,
            top_logprobs: Optional[int] = None,
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
            "min_p": 0.0,
            "min_tokens_to_keep": 1,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.0,
        }

        if top_k is not None:
            core_sampler_kwargs["top_k"] = top_k

        # Merge with additional sampler kwargs
        if sampler_kwargs:
            core_sampler_kwargs.update(sampler_kwargs)

        mlx_kwargs["sampler"] = make_sampler(**core_sampler_kwargs)

        # Handle special cases that need preprocessing
        handled_kwargs = set()

        if kwargs.get("repetition_penalty") is not None:
            from mlx_lm.sample_utils import make_logits_processors

            existing_processors = mlx_kwargs.get("logits_processors", [])
            new_processors = make_logits_processors(
                repetition_penalty=kwargs["repetition_penalty"]
            )
            mlx_kwargs["logits_processors"] = existing_processors + new_processors
            handled_kwargs.add("repetition_penalty")

        if kwargs.get("json_schema") is not None:
            from .outlines_logits_processor import OutlinesLogitsProcessor

            logits_processors = mlx_kwargs.get("logits_processors", [])
            logits_processors.append(
                OutlinesLogitsProcessor(self.tokenizer, kwargs["json_schema"])
            )
            mlx_kwargs["logits_processors"] = logits_processors
            handled_kwargs.add("json_schema")

        # Handle remaining valid kwargs
        for key, value in kwargs.items():
            if key not in handled_kwargs and value is not None:
                mlx_kwargs[key] = value

        return mlx_kwargs

    def _process_logprobs(
        self, response, top_k: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Process logprobs from MLX response.

        Args:
            response: MLX response object
            top_k: Number of top logprobs to include

        Returns:
            Processed logprobs dictionary or None
        """
        if not hasattr(response, "logprobs") or response.logprobs is None:
            return None

        current_token = response.token
        current_logprobs = response.logprobs

        token_str = self.tokenizer.decode([current_token])
        token_logprob = mx.clip(
            current_logprobs[current_token], a_min=-100, a_max=None
        ).item()
        token_bytes = token_str.encode("utf-8")

        token_info = {
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_bytes),
        }

        top_logprobs = []
        if top_k is not None:
            top_indices = mx.argpartition(-current_logprobs, kth=top_k - 1)[:top_k]
            top_probs = mx.clip(current_logprobs[top_indices], a_min=-100, a_max=None)

            for idx, logprob in zip(top_indices.tolist(), top_probs.tolist()):
                token = self.tokenizer.decode([idx])
                token_bytes = token.encode("utf-8")
                top_logprobs.append(
                    {"token": token, "logprob": logprob, "bytes": list(token_bytes)}
                )

        return {**token_info, "top_logprobs": top_logprobs}

    def _get_logprobs(
        self, response, top_logprobs: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Get logprobs from response if requested.

        Args:
            response: MLX response object
            top_logprobs: Number of top logprobs to include

        Returns:
            Processed logprobs dictionary or None
        """
        if top_logprobs is not None:
            return self._process_logprobs(response, top_logprobs)
        return None

    def _create_generation_stats(
        self, response, cached_tokens: int = 0
    ) -> GenerationStats:
        return GenerationStats(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.generation_tokens,
            prompt_tps=response.prompt_tps,
            generation_tps=response.generation_tps,
            peak_memory=response.peak_memory,
            cache_hit_tokens=cached_tokens,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        # Core generation parameters
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        # Logprobs parameter (MLX style)
        top_logprobs: Optional[int] = None,
        # Additional sampler parameters
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        # Template parameters
        template_kwargs: Optional[Dict[str, Any]] = None,
        # Control parameters
        enable_prompt_cache: bool = True,
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
            complete_text = ""
            final_result = None

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
                complete_text += result.text
                final_result = result

            if final_result is None:
                raise RuntimeError("No tokens generated")

            # Process in correct order: reasoning → tools → plain text
            reasoning = None
            tool_calls = None

            # Step 1: Process reasoning first (extract <think> tags)
            # Only extract reasoning when enabled
            enable_reasoning = bool(
                template_kwargs and template_kwargs.get("enable_thinking", True)
            )
            logger.info(
                f"enable_reasoning: {enable_reasoning}, complete_text: {complete_text[:100]}..."
            )
            old_enable_thinking = self.reasoning_decoder.enable_thinking
            self.reasoning_decoder.enable_thinking = enable_reasoning
            try:
                if enable_reasoning:
                    reasoning_result = self.reasoning_decoder.decode(complete_text)
                    if reasoning_result:
                        reasoning = reasoning_result.get("reasoning")
                        complete_text = reasoning_result.get("content", complete_text)
                        logger.debug(
                            f"Extracted reasoning: {reasoning is not None}, new content: {complete_text[:50]}..."
                        )
                    else:
                        logger.debug("No reasoning extracted")
                        reasoning = None
                else:
                    # When reasoning is disabled, don't extract reasoning content
                    reasoning = None
                    logger.debug("Reasoning disabled, removing think tags")
                    # Remove think tags even when reasoning is disabled
                    reasoning_result = self.reasoning_decoder.decode(complete_text)
                    if reasoning_result:
                        complete_text = reasoning_result.get("content", complete_text)
                    else:
                        complete_text = complete_text
            finally:
                self.reasoning_decoder.enable_thinking = old_enable_thinking

            # Step 2: Process tools from remaining content
            if tools:
                tool_calls = self.chat_tokenizer.decode(complete_text)

            # Step 3: Remaining content is plain text (already in complete_text)

            # Determine appropriate finish_reason
            finish_reason = final_result.finish_reason
            if tool_calls:
                finish_reason = "tools"

            # Return final result with all processing applied
            return GenerationResult(
                text=complete_text,
                token=final_result.token,
                finish_reason=finish_reason,
                stats=final_result.stats,
                tool_calls=tool_calls,
                reasoning=reasoning,
                logprobs=final_result.logprobs,
                from_draft=final_result.from_draft,
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
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        # Logprobs parameter (MLX style)
        top_logprobs: Optional[int] = None,
        # Additional sampler parameters
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        # Template parameters
        template_kwargs: Optional[Dict[str, Any]] = None,
        # Control parameters
        enable_prompt_cache: bool = True,
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
            # Determine processing flags from parameters
            enable_tools = bool(tools)
            enable_reasoning = bool(
                template_kwargs and template_kwargs.get("enable_thinking", True)
            )
            logger.info(f"enable_reasoning: {enable_reasoning}")

            # Prepare prompt
            prompt_str = self._prepare_prompt(messages, tools, template_kwargs)

            # Tokenize prompt
            tokenized_prompt = self.tokenizer.encode(prompt_str)

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
                top_logprobs=top_logprobs,
                **kwargs,
            )

            # Add cache to kwargs if available
            if enable_prompt_cache and self.prompt_cache.cache:
                mlx_kwargs["prompt_cache"] = self.prompt_cache.cache

            # Stream generation
            generated_tokens = []
            completion_text = ""

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
                logprobs = self._get_logprobs(response, top_logprobs)

                # Track completion text for reasoning processing
                current_text = response.text
                completion_text += current_text

                # Process in correct order for streaming: reasoning → tools → plain text
                current_reasoning = None
                current_tool_calls = None

                # Step 1: Process reasoning for streaming if enabled
                if enable_reasoning:
                    reasoning_result = self.reasoning_decoder.stream_decode(
                        current_text
                    )
                    if reasoning_result:
                        # Use processed content if available
                        if reasoning_result.get("delta_content") is not None:
                            current_text = reasoning_result.get("delta_content")
                        # Extract reasoning data if available
                        current_reasoning = reasoning_result.get("delta_reasoning")

                # Step 2: For tools, we need to process the accumulated text
                # since tools usually require complete structures
                # if enable_tools and tools:
                #     # Try to decode tools from the accumulated completion text
                #     current_tool_calls = self.chat_tokenizer.decode(completion_text)

                # Step 3: current_text now contains the processed content
                yield GenerationResult(
                    text=current_text,
                    token=response.token,
                    finish_reason=response.finish_reason,
                    stats=self._create_generation_stats(response, cached_tokens),
                    tool_calls=current_tool_calls,
                    reasoning=current_reasoning,
                    logprobs=logprobs,
                    from_draft=response.from_draft,
                )

            # Extend cache with generated tokens if caching is enabled
            if enable_prompt_cache and generated_tokens:
                self.prompt_cache.extend_completion_cache(generated_tokens)

        except Exception as e:
            logger.error(f"Error during stream generation: {e}")
            raise RuntimeError(f"Stream generation failed: {e}")
