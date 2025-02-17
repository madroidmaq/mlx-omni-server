import time
import uuid
from typing import Any, Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import GenerationResponse, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...utils.logger import logger
from ..schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)
from ..text_models import BaseTextModel, GenerateResult, GenerationParams
from .outlines_logits_processor import OutlinesLogitsProcessor
from .prompt_cache import PromptCache
from .stop_tokens_checker import StopTokensChecker
from .tools.chat_tokenizer import ChatTokenizer
from .tools.reasoning_decoder import ReasoningDecoder


class MLXModel(BaseTextModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(
        self,
        model_id: str,
        model: nn.Module,
        tokenizer: ChatTokenizer,
        draft_model=None,
    ):
        self._model_id = model_id
        self._model: nn.Module = model
        self._draft_model = draft_model
        self._default_max_tokens = 2048
        self._default_temperature = 1.0
        self._default_top_p = 1.0
        self._default_top_k = -1
        self._chat_tokenizer = tokenizer
        self._prompt_cache = PromptCache()
        self._prompt_cache_tokens_count = 0
        self._reasoning_decoder = ReasoningDecoder(tokenizer)
        logger.info(f"Initialized MLXModel with model_id: {model_id}")

    def _get_generation_params(
        self, request: ChatCompletionRequest
    ) -> GenerationParams:
        params = request.get_extra_params()

        # All params declare in `make_sampler`
        sampler_params = {
            "top_k",
            "min_tokens_to_keep",
            "min_p",
            "xtc_probability",
            "xtc_threshold",
            "xtc_special_tokens",
        }
        # Knowned params using in model config
        model_params = {
            "adapter_path",
            "draft_model",
            # Additional config for `apply_chat_template`
            "chat_template_config",
        }
        # Quick template params, same param will be overrided by `chat_template_config`
        template_params = {
            # Qwen3
            "enable_thinking",
            "thinking_budget",
            # Claude
            "thinking",
            # Gemini
            "thinkingConfig",
            # Grok
            "reasoning_effort",
            # Others
            "reasoning",
        }

        sampler_kwargs = {}
        model_kwargs = {}
        generate_kwargs = {}
        template_kwargs = {}

        for key, value in params.items():
            if key in sampler_params:
                sampler_kwargs[key] = value
            elif key in model_params:
                model_kwargs[key] = value
            elif key in template_params:
                template_kwargs[key] = value
            else:
                generate_kwargs[key] = value

        return {
            "sampler_kwargs": sampler_kwargs,
            "model_kwargs": model_kwargs,
            "generate_kwargs": generate_kwargs,
            "template_kwargs": template_kwargs,
        }

    def _process_logprobs(
        self,
        tokenizer: TokenizerWrapper,
        response: GenerationResponse,
        top_k: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Process logprobs information from generation response to match OpenAI format"""
        current_token = response.token
        current_logprobs = response.logprobs

        # Get current token info
        token_str = tokenizer.decode([current_token])
        token_logprob = current_logprobs[current_token].item()
        token_bytes = token_str.encode("utf-8")

        # Base token info
        token_info = {
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_bytes),
        }

        # Process top logprobs
        top_logprobs = []
        if top_k is not None:
            # Get indices of top_k tokens
            top_indices = mx.argpartition(-current_logprobs, kth=top_k - 1)[:top_k]
            top_probs = current_logprobs[top_indices]

            # Create detailed token information for each top token
            for idx, logprob in zip(top_indices.tolist(), top_probs.tolist()):
                token = tokenizer.decode([idx])
                token_bytes = token.encode("utf-8")
                top_logprobs.append(
                    {"token": token, "logprob": logprob, "bytes": list(token_bytes)}
                )

        return {**token_info, "top_logprobs": top_logprobs}

    def _prepare_generation(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[Any, StopTokensChecker | None, dict[str, Any]]:
        """Prepare all necessary components for generation.

        This function handles parameter processing, tokenizer setup, prompt encoding,
        sampler creation, and other preparation work for text generation.

        Args:
            request: The chat completion request containing generation parameters

        Returns:
            A tuple containing tokenizer, processed prompt, stop checker, and generation kwargs
        """
        # Process parameters from request
        params = self._get_generation_params(request)

        model_kwargs = params.get("model_kwargs", {})
        logger.debug(f"Model kwargs: {model_kwargs}")

        template_kwargs = params.get("template_kwargs") | model_kwargs.get(
            "chat_template_config", {}
        )
        logger.debug(f"Chat Template kwargs: {template_kwargs}")

        # Prepare generation kwargs
        generate_kwargs = params.get("generate_kwargs", {})

        # Prepare sampler parameters
        sampler_kwargs = {
            "temp": (
                self._default_temperature
                if request.temperature is None
                else request.temperature
            ),
            "top_p": (self._default_top_p if request.top_p is None else request.top_p),
            "min_p": 0.0,
            "min_tokens_to_keep": 1,
            "top_k": self._default_top_k,
        } | params.get("sampler_kwargs", {})

        logger.debug(f"Sampler kwargs: {sampler_kwargs}")

        # Create sampler and add to generate_kwargs
        generate_kwargs["sampler"] = make_sampler(**sampler_kwargs)

        # Encode prompt with chat template
        prompt = self._chat_tokenizer.encode(
            messages=request.messages,
            tools=request.tools,
            **template_kwargs,
        )
        logger.debug(f"Encoded prompt:\n{prompt}")

        enable_thinking = template_kwargs.get("enable_thinking", True)
        self._reasoning_decoder.enable_thinking = enable_thinking
        if enable_thinking:
            self._reasoning_decoder.set_thinking_prefix(True)
            if prompt.endswith(f"<{self._reasoning_decoder.thinking_tag}>"):
                self._reasoning_decoder.set_thinking_prefix(True)
            else:
                self._reasoning_decoder.set_thinking_prefix(False)

        # Get tokenizer
        tokenizer = self._chat_tokenizer.tokenizer

        # Process prompt cache
        tokenized_prompt = tokenizer.encode(prompt)
        processed_prompt = self._prompt_cache.get_prompt_cache(
            self._model_id, self._model, tokenized_prompt
        )
        generate_kwargs["prompt_cache"] = self._prompt_cache.cache
        logger.debug(
            f"Using {self._prompt_cache.cached_token_count} cached tokens out of {len(tokenized_prompt)} total tokens"
        )

        # Setup stop tokens checker if needed
        stop_checker = None
        if request.stop:
            stop_checker = StopTokensChecker(
                stop_words=request.stop,
                tokenizer=tokenizer,
            )

        # Setup logits processors
        if request.response_format and request.response_format.json_schema:
            generate_kwargs["logits_processors"] = [
                OutlinesLogitsProcessor(
                    self._chat_tokenizer.tokenizer, request.response_format
                )
            ]
        elif request.presence_penalty:
            generate_kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=request.presence_penalty
            )

        # Calculate max tokens for completion
        generate_kwargs["max_tokens"] = (
            request.max_completion_tokens
            or request.max_tokens
            or self._default_max_tokens
        )

        return processed_prompt, stop_checker, generate_kwargs

    def _stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[GenerateResult, None, None]:
        try:
            # Get tokenizer
            tokenizer = self._chat_tokenizer.tokenizer

            # Prepare all generation components
            processed_prompt, stop_checker, generate_kwargs = self._prepare_generation(
                request
            )

            current_tokens = []
            last_text = ""

            for response in stream_generate(
                model=self._model,
                tokenizer=tokenizer,
                prompt=processed_prompt,
                draft_model=self._draft_model,
                **generate_kwargs,
            ):
                if response.finish_reason is not None:
                    break

                current_tokens.append(response.token)

                logprobs = None
                if request.logprobs:
                    logprobs = self._process_logprobs(
                        tokenizer, response, request.top_logprobs
                    )

                finish_reason = response.finish_reason
                should_trim = False
                if request.stop and stop_checker:
                    stop_condition = stop_checker.check_stop_condition(current_tokens)
                    if stop_condition.stop_met:
                        finish_reason = "stop"
                        if stop_condition.trim_length > 0:
                            current_tokens = current_tokens[
                                : -stop_condition.trim_length
                            ]
                            should_trim = True

                text = tokenizer.decode(current_tokens)
                delta_text = text[len(last_text) :]

                if delta_text or should_trim:
                    yield GenerateResult(
                        text=delta_text,
                        token=response.token,
                        finish_reason=finish_reason,
                        prompt_tokens=response.prompt_tokens,
                        generation_tokens=response.generation_tokens,
                        logprobs=logprobs,
                    )
                    last_text = text

                if should_trim:
                    break

            self._prompt_cache_tokens_count = self._prompt_cache.cached_token_count
            logger.debug(
                f"The generation is completed, with a total of {self._prompt_cache_tokens_count} tokens cached."
            )
            self._prompt_cache.extend_completion_cache(current_tokens)
        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        try:
            completion = ""
            logprobs_result_list = []
            current_tokens = []
            finish_reason = "stop"
            result = None

            for result in self._stream_generate(request=request):
                current_tokens.append(result.token)
                completion = self._chat_tokenizer.tokenizer.decode(current_tokens)

                if request.logprobs:
                    logprobs_result_list.append(result.logprobs)

                if result.finish_reason:
                    finish_reason = result.finish_reason

            if result is None:
                raise RuntimeError("No tokens generated")

            logger.debug(f"Model Response:\n{completion}")
            reasoning: str | None = None  # avoid UnboundLocalError
            enable_thinking = self._reasoning_decoder.enable_thinking
            if enable_thinking:
                reasoning_result = self._reasoning_decoder.decode(completion)
                if reasoning_result:
                    logger.debug(f"Reasoning result:\n{reasoning_result}")
                    completion = reasoning_result.get("content")
                    reasoning = reasoning_result.get("reasoning") or None

            if request.tools:
                message = self._chat_tokenizer.decode(completion)
            else:
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=completion,
                    reasoning=reasoning,
                )

            cached_tokens = self._prompt_cache_tokens_count
            logger.debug(f"Generate response with {cached_tokens} cached tokens")

            prompt_tokens_details = None
            if cached_tokens > 0:
                from ..schema import PromptTokensDetails

                prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason=(
                            "tool_calls" if message.tool_calls else finish_reason
                        ),
                        logprobs=(
                            {"content": logprobs_result_list}
                            if logprobs_result_list
                            else None
                        ),
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens + cached_tokens,
                    completion_tokens=result.generation_tokens,
                    total_tokens=result.prompt_tokens
                    + result.generation_tokens
                    + cached_tokens,
                    prompt_tokens_details=prompt_tokens_details,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

            completion = ""
            for result in self._stream_generate(request=request):
                created = int(time.time())
                completion += result.text

                message = None
                enable_thinking = self._reasoning_decoder.enable_thinking
                if enable_thinking:
                    reasoning_result = self._reasoning_decoder.stream_decode(
                        result.text
                    )
                    if reasoning_result:
                        logger.debug(f"Reasoning result:\n{reasoning_result}")
                        delta_content = reasoning_result.get("delta_content")
                        delta_reasoning = (
                            reasoning_result.get("delta_reasoning") or None
                        )
                        if delta_content:
                            message = ChatMessage(
                                role=Role.ASSISTANT,
                                content=delta_content or result.text,
                            )
                        elif delta_reasoning:
                            message = ChatMessage(
                                role=Role.ASSISTANT, reasoning=delta_reasoning
                            )

                if message is None:
                    message = ChatMessage(role=Role.ASSISTANT, content=result.text)

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=message,
                            finish_reason=result.finish_reason,
                            logprobs=result.logprobs,
                        )
                    ],
                )

            if request.stream_options and request.stream_options.include_usage:
                created = int(time.time())
                cached_tokens = self._prompt_cache_tokens_count
                logger.debug(f"Stream response with {cached_tokens} cached tokens")

                prompt_tokens_details = None
                if cached_tokens > 0:
                    from ..schema import PromptTokensDetails

                    prompt_tokens_details = PromptTokensDetails(
                        cached_tokens=cached_tokens
                    )

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT),
                            finish_reason=None,
                            logprobs=None,
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.prompt_tokens + cached_tokens,
                        completion_tokens=result.generation_tokens,
                        total_tokens=result.prompt_tokens
                        + result.generation_tokens
                        + cached_tokens,
                        prompt_tokens_details=prompt_tokens_details,
                    ),
                )

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise
