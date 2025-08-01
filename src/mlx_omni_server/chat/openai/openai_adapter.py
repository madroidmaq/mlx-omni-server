import time
import uuid
from typing import Generator

from mlx_omni_server.chat.mlx.chat_generator import DEFAULT_MAX_TOKENS, ChatGenerator
from mlx_omni_server.chat.openai.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)
from mlx_omni_server.utils.logger import logger


class OpenAIAdapter:
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(
        self,
        wrapper: ChatGenerator,
    ):
        """Initialize MLXModel with wrapper object.

        Args:
            wrapper: ChatGenerator instance (cached and ready to use)
        """
        self._default_max_tokens = DEFAULT_MAX_TOKENS
        self._generate_wrapper = wrapper

    def _prepare_generation_params(self, request: ChatCompletionRequest) -> dict:
        """Prepare common parameters for both generate and stream_generate."""
        max_tokens = (
            request.max_completion_tokens
            or request.max_tokens
            or self._default_max_tokens
        )

        # Extract parameters from request and extra params
        extra_params = request.get_extra_params()
        extra_body = extra_params.get("extra_body", {})

        # Prepare sampler configuration
        sampler_config = {
            "temp": 1.0 if request.temperature is None else request.temperature,
            "top_p": 1.0 if request.top_p is None else request.top_p,
            "top_k": extra_body.get("top_k", 0),
        }

        # Add additional sampler parameters from extra_body
        if extra_body.get("min_p") is not None:
            sampler_config["min_p"] = extra_body.get("min_p")
        if extra_body.get("min_tokens_to_keep") is not None:
            sampler_config["min_tokens_to_keep"] = extra_body.get("min_tokens_to_keep")
        if extra_body.get("xtc_probability") is not None:
            sampler_config["xtc_probability"] = extra_body.get("xtc_probability")
        if extra_body.get("xtc_threshold") is not None:
            sampler_config["xtc_threshold"] = extra_body.get("xtc_threshold")

        # Prepare template parameters - include both extra_body and direct extra params
        template_kwargs = dict(extra_body)

        # Handle direct extra parameters (for backward compatibility)
        for key in ["enable_thinking"]:
            if key in extra_params:
                template_kwargs[key] = extra_params[key]

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
        logger.info(f"template_kwargs: {template_kwargs}")

        json_schema = None
        if request.response_format and request.response_format.json_schema:
            json_schema = request.response_format.json_schema.schema_def

        return {
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "sampler": sampler_config,
            "top_logprobs": request.top_logprobs if request.logprobs else None,
            "template_kwargs": template_kwargs,
            "enable_prompt_cache": True,
            "repetition_penalty": request.presence_penalty,
            "json_schema": json_schema,
        }

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate complete response using the wrapper."""
        try:
            # Prepare parameters
            params = self._prepare_generation_params(request)

            # Directly use wrapper's generate method for complete response
            result = self._generate_wrapper.generate(**params)

            logger.debug(f"Model Response:\n{result.content.text}")

            # Use reasoning from the wrapper's result
            final_content = result.content.text
            reasoning_content = result.content.reasoning

            # Use wrapper's chat tokenizer for tool processing
            if request.tools:
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=final_content,
                    tool_calls=result.content.tool_calls,
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
                        logprobs=result.logprobs,
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

    def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Stream generate OpenAI-compatible chunks."""
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

            # Prepare parameters
            params = self._prepare_generation_params(request)

            result = None
            for chunk in self._generate_wrapper.generate_stream(**params):
                created = int(time.time())

                # TODO: support streaming tools parse
                # For streaming, we need to get the appropriate delta content
                if chunk.content.text_delta:
                    content = chunk.content.text_delta
                elif chunk.content.reasoning_delta:
                    content = chunk.content.reasoning_delta
                else:
                    content = ""

                message = ChatMessage(role=Role.ASSISTANT, content=content)

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=message,
                            finish_reason=chunk.finish_reason or "stop",
                            logprobs=chunk.logprobs,
                        )
                    ],
                )
                result = chunk

            if (
                request.stream_options
                and request.stream_options.include_usage
                and result is not None
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
