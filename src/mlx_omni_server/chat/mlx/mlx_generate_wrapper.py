"""MLX Generate Wrapper - Core abstraction layer over mlx-lm."""

from dataclasses import replace
from typing import Any, Dict, Generator, List, Optional

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from ...utils.logger import logger
from .core_types import (
    ChatTemplateConfig,
    GenerationResult,
    GenerationStats,
    MLXGenerateConfig,
    SamplerConfig,
)
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
        template_config: Optional[ChatTemplateConfig] = None,
    ) -> str:
        """Prepare prompt using chat tokenizer.

        Args:
            messages: Chat messages in standard format (dictionaries)
            tools: Optional tools for function calling
            template_config: Chat template configuration

        Returns:
            Encoded prompt string
        """
        if tools:
            logger.debug(f"Prepared {len(tools)} tools for encoding")

        # Extract template kwargs from config
        template_kwargs = {}
        if template_config:
            template_kwargs.update(template_config.template_kwargs or {})

            # Add thinking-related parameters if present
            thinking_params = ["enable_thinking", "thinking_budget", "reasoning_effort"]
            for param in thinking_params:
                value = getattr(template_config, param, None)
                if value is not None:
                    template_kwargs[param] = value

        prompt = self.chat_tokenizer.encode(
            messages=messages, tools=tools, **template_kwargs
        )

        logger.debug(f"Encoded prompt: {prompt}")
        return prompt

    def _create_mlx_kwargs(
        self, sampler_config: SamplerConfig, generate_config: MLXGenerateConfig
    ) -> Dict[str, Any]:
        """Convert configs to mlx-lm compatible kwargs.

        Args:
            sampler_config: Sampler configuration
            generate_config: Generation configuration

        Returns:
            Dictionary of kwargs for mlx-lm generate functions
        """
        # Core MLX parameters
        mlx_kwargs = {
            "max_tokens": generate_config.max_tokens,
        }

        # Create sampler
        sampler_kwargs = {
            "temp": sampler_config.temperature,
            "top_p": sampler_config.top_p,
            "min_p": sampler_config.min_p,
            "min_tokens_to_keep": sampler_config.min_tokens_to_keep,
            "xtc_probability": sampler_config.xtc_probability,
            "xtc_threshold": sampler_config.xtc_threshold,
        }

        if sampler_config.top_k is not None:
            sampler_kwargs["top_k"] = sampler_config.top_k

        mlx_kwargs["sampler"] = make_sampler(**sampler_kwargs)

        # Performance options
        if generate_config.max_kv_size is not None:
            mlx_kwargs["max_kv_size"] = generate_config.max_kv_size
        if generate_config.kv_bits is not None:
            mlx_kwargs["kv_bits"] = generate_config.kv_bits
            mlx_kwargs["kv_group_size"] = generate_config.kv_group_size
            mlx_kwargs["quantized_kv_start"] = generate_config.quantized_kv_start

        # Generation control
        if generate_config.repetition_penalty is not None:
            from mlx_lm.sample_utils import make_logits_processors

            mlx_kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=generate_config.repetition_penalty
            )

        if generate_config.seed is not None:
            # Note: MLX stream_generate doesn't support seed parameter
            # This would need to be set before generation starts
            pass

        # Draft model
        if self.model_cache.draft_model is not None:
            mlx_kwargs["draft_model"] = self.model_cache.draft_model
            mlx_kwargs["num_draft_tokens"] = generate_config.num_draft_tokens

        # JSON schema
        if generate_config.json_schema is not None:
            from .outlines_logits_processor import OutlinesLogitsProcessor

            logits_processors = mlx_kwargs.get("logits_processors", [])
            logits_processors.append(
                OutlinesLogitsProcessor(self.tokenizer, generate_config.json_schema)
            )
            mlx_kwargs["logits_processors"] = logits_processors

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

    def _prepare_configs(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        top_logprobs: Optional[int],
        sampler_config: Optional[SamplerConfig],
        generate_config: Optional[MLXGenerateConfig],
    ) -> tuple[SamplerConfig, MLXGenerateConfig]:
        """Prepare sampler and generation configurations.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling value
            top_k: Top-k sampling value
            top_logprobs: Number of top logprobs to include
            sampler_config: Optional sampler configuration
            generate_config: Optional generation configuration

        Returns:
            Tuple of (sampler_config, generate_config)
        """
        # Use provided configs or create defaults with common parameters
        if sampler_config is None:
            sampler_config = SamplerConfig(
                temperature=temperature, top_p=top_p, top_k=top_k
            )

        if generate_config is None:
            generate_config = MLXGenerateConfig(
                max_tokens=max_tokens, top_logprobs=top_logprobs
            )
        else:
            # Override config values with function parameters using dataclasses.replace
            generate_config = replace(
                generate_config,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
            )

        return sampler_config, generate_config

    def _setup_reasoning_decoder(
        self, template_config: Optional[ChatTemplateConfig], prompt_str: str
    ):
        """Setup reasoning decoder for generation.

        Args:
            template_config: Chat template configuration
            prompt_str: The prepared prompt string
        """
        template_kwargs = template_config.template_kwargs if template_config else {}
        enable_thinking = template_kwargs.get("enable_thinking", True)
        self.reasoning_decoder.enable_thinking = enable_thinking

        if enable_thinking:
            self.reasoning_decoder.set_thinking_prefix(True)
            if prompt_str.endswith(f"<{self.reasoning_decoder.thinking_tag}>"):
                self.reasoning_decoder.set_thinking_prefix(True)
            else:
                self.reasoning_decoder.set_thinking_prefix(False)

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
        # Common parameters promoted to function arguments
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        # Logprobs parameter (MLX style)
        top_logprobs: Optional[int] = None,
        # Advanced configurations
        sampler_config: Optional[SamplerConfig] = None,
        generate_config: Optional[MLXGenerateConfig] = None,
        template_config: Optional[ChatTemplateConfig] = None,
        # Control parameters
        enable_prompt_cache: bool = True,
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
            sampler_config: Advanced sampler configuration
            generate_config: MLX generation configuration
            template_config: Chat template configuration
            enable_prompt_cache: Enable prompt caching

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
                sampler_config,
                generate_config,
                template_config,
                enable_prompt_cache,
            ):
                complete_text += result.text
                final_result = result

            if final_result is None:
                raise RuntimeError("No tokens generated")

            # Process in correct order: reasoning → tools → plain text
            reasoning = None
            tool_calls = None

            # Step 1: Process reasoning first (extract <think> tags)
            # Always extract reasoning when <think> tags are present
            old_enable_thinking = self.reasoning_decoder.enable_thinking
            self.reasoning_decoder.enable_thinking = True
            try:
                reasoning_result = self.reasoning_decoder.decode(complete_text)
                if reasoning_result:
                    reasoning = reasoning_result.get("reasoning")
                    complete_text = reasoning_result.get("content", complete_text)
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
        # Common parameters promoted to function arguments
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        # Logprobs parameter (MLX style)
        top_logprobs: Optional[int] = None,
        # Advanced configurations
        sampler_config: Optional[SamplerConfig] = None,
        generate_config: Optional[MLXGenerateConfig] = None,
        template_config: Optional[ChatTemplateConfig] = None,
        # Control parameters
        enable_prompt_cache: bool = True,
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
            sampler_config: Advanced sampler configuration
            generate_config: MLX generation configuration
            template_config: Chat template configuration
            enable_prompt_cache: Enable prompt caching

        Yields:
            Streaming generation results
        """
        try:
            # Prepare configurations
            sampler_config, generate_config = self._prepare_configs(
                max_tokens,
                temperature,
                top_p,
                top_k,
                top_logprobs,
                sampler_config,
                generate_config,
            )

            # Determine processing flags from parameters
            enable_tools = bool(tools)
            enable_reasoning = bool(template_config and template_config.enable_thinking)

            # Prepare prompt
            prompt_str = self._prepare_prompt(messages, tools, template_config)

            # Tokenize prompt
            tokenized_prompt = self.tokenizer.encode(prompt_str)

            # Process cache if enabled
            processed_prompt = tokenized_prompt
            cached_tokens = 0

            if enable_prompt_cache:
                processed_prompt, cached_tokens = self.prompt_cache.get_prompt_cache(
                    self.model_cache, tokenized_prompt
                )

            # Setup reasoning decoder if enabled
            if enable_reasoning:
                self._setup_reasoning_decoder(template_config, prompt_str)

            # Create MLX kwargs
            mlx_kwargs = self._create_mlx_kwargs(sampler_config, generate_config)

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
                logprobs = self._get_logprobs(response, generate_config.top_logprobs)

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
                if enable_tools and tools:
                    # Try to decode tools from the accumulated completion text
                    current_tool_calls = self.chat_tokenizer.decode(completion_text)
                    # For streaming, we might want to show partial tool calls
                    # but keep the original current_text for now

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
