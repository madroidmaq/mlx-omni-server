import time
import uuid
from typing import Generator

from mlx_omni_server.chat.mlx.core_types import GenerationResult
from mlx_omni_server.chat.mlx.mlx_generate_wrapper import MLXGenerateWrapper
from mlx_omni_server.chat.mlx.model_types import MlxModelCache
from mlx_omni_server.chat.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)
from mlx_omni_server.chat.text_models import BaseTextModel
from mlx_omni_server.utils.logger import logger


class OpenAIAdapter(BaseTextModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(
        self,
        model_cache: MlxModelCache,
    ):
        """Initialize MLXModel with model cache object.

        Args:
            model_cache: MlxModelCache object containing models and tokenizers
        """
        self._default_max_tokens = 2048
        self._generate_wrapper = MLXGenerateWrapper(model_cache)

    def _stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[GenerationResult, None, None]:
        """Stream generate using the wrapper with OpenAI format conversion."""
        try:
            # Extract basic parameters from OpenAI request
            max_tokens = (
                request.max_completion_tokens
                or request.max_tokens
                or self._default_max_tokens
            )

            # Extract parameters from request and extra params
            extra_params = request.get_extra_params()
            extra_body = extra_params.get("extra_body", {})

            # Prepare sampler parameters
            sampler_kwargs = {
                "min_p": extra_body.get("min_p", 0.0),
                "min_tokens_to_keep": extra_body.get("min_tokens_to_keep", 1),
                "xtc_probability": extra_body.get("xtc_probability", 0.0),
                "xtc_threshold": extra_body.get("xtc_threshold", 0.0),
            }
            if extra_body.get("top_k") is not None:
                sampler_kwargs["top_k"] = extra_body.get("top_k")

            # Prepare template parameters
            template_kwargs = dict(extra_body)
            template_kwargs.setdefault("enable_thinking", True)

            # Convert messages to dict format
            messages = [
                {
                    "role": (
                        msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    ),
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                }
                for msg in request.messages
            ]

            # Convert tools to dict format
            tools = None
            if request.tools:
                tools = [
                    tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
                    for tool in request.tools
                ]

            logger.info(f"messages: {messages}")

            # Delegate to wrapper
            for result in self._generate_wrapper.stream_generate(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=1.0 if request.temperature is None else request.temperature,
                top_p=1.0 if request.top_p is None else request.top_p,
                top_k=extra_body.get("top_k", 0),
                top_logprobs=request.top_logprobs if request.logprobs else None,
                sampler_kwargs=sampler_kwargs,
                template_kwargs=template_kwargs,
                enable_prompt_cache=True,
                repetition_penalty=request.presence_penalty,
                json_schema=(
                    request.response_format.json_schema
                    if request.response_format
                    else None
                ),
                # max_kv_size=extra_body.get("max_kv_size"),
                # kv_bits=extra_body.get("kv_bits"),
                # kv_group_size=extra_body.get("kv_group_size", 64),
                # quantized_kv_start=extra_body.get("quantized_kv_start", 0),
                # num_draft_tokens=extra_body.get("num_draft_tokens", 3),
            ):
                yield result

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate complete response using the wrapper."""
        try:
            # Collect all tokens for complete response
            completion = ""
            logprobs_result_list = []

            for stream_result in self._stream_generate(request=request):
                completion += stream_result.text
                if request.logprobs:
                    logprobs_result_list.append(stream_result.logprobs)
                result = stream_result

            if result is None:
                raise RuntimeError("No tokens generated")

            logger.debug(f"Model Response:\n{completion}")

            # Use reasoning from the wrapper's result instead of re-processing
            # The wrapper has already handled reasoning extraction based on enable_thinking
            final_content = completion
            reasoning_content = result.reasoning

            # Use wrapper's chat tokenizer for tool processing
            if request.tools:
                # Pass the content *after* reasoning has been stripped
                tool_calls = self._generate_wrapper.chat_tokenizer.decode(final_content)
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=final_content,
                    tool_calls=tool_calls,
                    reasoning=reasoning_content,
                )
            else:
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=final_content,
                    reasoning=reasoning_content,
                )

            # Use cached tokens from wrapper stats
            cached_tokens = result.stats.cache_hit_tokens
            logger.debug(f"Generate response with {cached_tokens} cached tokens")

            prompt_tokens_details = None
            if cached_tokens > 0:
                from .schema import PromptTokensDetails

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
                            "tool_calls"
                            if message.tool_calls
                            else (result.finish_reason or "stop")
                        ),
                        logprobs=(
                            {"content": logprobs_result_list}
                            if logprobs_result_list
                            else None
                        ),
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.stats.prompt_tokens + cached_tokens,
                    completion_tokens=result.stats.completion_tokens,
                    total_tokens=result.stats.prompt_tokens
                    + result.stats.completion_tokens
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
        """Stream generate OpenAI-compatible chunks."""
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

            completion = ""
            result = None
            for result in self._stream_generate(request=request):
                created = int(time.time())
                completion += result.text

                # Use wrapper's chat tokenizer for tool processing
                if request.tools:
                    message = self._generate_wrapper.chat_tokenizer.decode(result.text)
                else:
                    # For streaming, we need to accumulate reasoning content
                    # In streaming mode, reasoning is handled by the decoder
                    message = ChatMessage(role=Role.ASSISTANT, content=result.text)

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=message,
                            finish_reason=result.finish_reason or "stop",
                            logprobs=result.logprobs,
                        )
                    ],
                )

            if (
                request.stream_options
                and request.stream_options.include_usage
                and result
            ):
                created = int(time.time())
                cached_tokens = result.stats.cache_hit_tokens
                logger.debug(f"Stream response with {cached_tokens} cached tokens")

                prompt_tokens_details = None
                if cached_tokens > 0:
                    from .schema import PromptTokensDetails

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
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.stats.prompt_tokens + cached_tokens,
                        completion_tokens=result.stats.completion_tokens,
                        total_tokens=result.stats.prompt_tokens
                        + result.stats.completion_tokens
                        + cached_tokens,
                        prompt_tokens_details=prompt_tokens_details,
                    ),
                )

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise
