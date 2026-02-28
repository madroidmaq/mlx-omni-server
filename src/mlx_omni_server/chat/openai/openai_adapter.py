import json
import time
import uuid
from typing import Any, Generator, List, Optional, Tuple

from mlx_omni_server.chat.mlx.chat_generator import (
    DEFAULT_MAX_TOKENS,
    ChatGenerator,
)
from mlx_omni_server.chat.mlx.core_types import ToolCall as CoreToolCall
from mlx_omni_server.chat.openai.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
    ToolCall,
)
from mlx_omni_server.utils.logger import logger

# Tool call XML markers used by models like Qwen3-Coder
TOOL_CALL_MARKERS = ["<tool_call>", "<function="]
# Buffer size for detecting tool call markers across chunk boundaries
TOOL_CALL_BUFFER_SIZE = 20


def _find_tool_call_marker_position(text: str) -> int:
    """Find the earliest position of a tool call marker in text.

    Args:
        text: The text to search for tool call markers

    Returns:
        Position of earliest marker, or -1 if no marker found
    """
    earliest_pos = -1
    for marker in TOOL_CALL_MARKERS:
        pos = text.find(marker)
        if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
            earliest_pos = pos
    return earliest_pos


def _has_tool_call_marker(text: str) -> bool:
    """Check if text contains any tool call marker.

    Args:
        text: The text to check

    Returns:
        True if any tool call marker is found
    """
    return any(marker in text for marker in TOOL_CALL_MARKERS)


def _convert_tool_calls(
    core_tool_calls: Optional[List[CoreToolCall]],
    deduplicate: bool = True,
) -> Optional[List[ToolCall]]:
    """Convert internal ToolCall format to OpenAI-compatible format.

    Args:
        core_tool_calls: List of tool calls from the model parser (core_types.ToolCall)
        deduplicate: If True, remove duplicate tool calls (same name and arguments)

    Returns:
        List of OpenAI-compatible tool calls (schema.ToolCall), or None if input is None
    """
    if not core_tool_calls:
        return None

    # Deduplicate tool calls based on name and arguments
    if deduplicate:
        seen = set()
        unique_calls = []
        for tc in core_tool_calls:
            # Create a hashable key using JSON serialization (handles nested structures)
            args_json = json.dumps(tc.arguments, sort_keys=True) if tc.arguments else ""
            key = (tc.name, args_json)
            if key not in seen:
                seen.add(key)
                unique_calls.append(tc)
            else:
                logger.debug(f"Deduplicated duplicate tool call: {tc.name}({tc.arguments})")
        core_tool_calls = unique_calls

    return [
        ToolCall.from_llama_output(
            name=tc.name,
            parameters=tc.arguments,
            call_id=tc.id,
            index=i,
        )
        for i, tc in enumerate(core_tool_calls)
    ]


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
        max_tokens = request.max_completion_tokens or request.max_tokens or self._default_max_tokens

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
                "role": (msg.role.value if hasattr(msg.role, "value") else str(msg.role)),
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
                **({"tool_calls": [
                    tc.model_dump(mode="json", exclude_none=True) 
                    if hasattr(tc, "model_dump") else tc 
                    for tc in msg.tool_calls
                ]} if msg.tool_calls else {}),
                **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
            }
            for msg in request.messages
        ]

        # Convert tools to dict format
        tools = None
        if request.tools:
            tools = [
                tool.model_dump(mode='json') if hasattr(tool, "model_dump") else dict(tool)
                for tool in request.tools
            ]

        logger.info(f"messages: {messages}")
        logger.info(f"template_kwargs: {template_kwargs}")
        logger.info(f"tools: {tools}")

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

    def _filter_stream_content(
        self,
        content: str,
        pending_buffer: str,
        in_tool_call: bool,
    ) -> Tuple[str, str, bool]:
        """Filter tool call XML from streaming content.

        Buffers incoming content to detect tool call markers that may span
        chunk boundaries. When a tool call marker is detected, stops streaming
        content to avoid displaying raw XML to users.

        Args:
            content: New content from the current chunk
            pending_buffer: Buffer of content not yet streamed
            in_tool_call: Whether we're currently inside a tool call block

        Returns:
            Tuple of (content_to_stream, updated_buffer, updated_in_tool_call)
        """
        pending_buffer += content

        if in_tool_call:
            # Inside tool call, don't stream anything
            return "", "", True

        # Look for tool call markers
        marker_pos = _find_tool_call_marker_position(pending_buffer)

        if marker_pos != -1:
            # Found marker - stream content before it, then enter tool call mode
            stream_content = pending_buffer[:marker_pos]
            logger.debug("Detected tool call start, switching to buffer mode")
            return stream_content, "", True

        if len(pending_buffer) > TOOL_CALL_BUFFER_SIZE:
            # No marker detected, safe to stream most of the buffer
            # Keep last N chars in case tag spans chunks
            stream_content = pending_buffer[:-TOOL_CALL_BUFFER_SIZE]
            return stream_content, pending_buffer[-TOOL_CALL_BUFFER_SIZE:], False

        # Buffer too small, wait for more content
        return "", pending_buffer, False

    def _parse_stream_tool_calls(
        self, accumulated_text: str
    ) -> Tuple[Optional[List[ToolCall]], str]:
        """Parse tool calls from accumulated streaming text.

        Args:
            accumulated_text: Full text accumulated during streaming

        Returns:
            Tuple of (tool_calls or None, finish_reason)
        """
        text_preview = (
            accumulated_text[:500] + "..." if len(accumulated_text) > 500 else accumulated_text
        )
        logger.info(f"Stream complete. Parsing for tool calls. Text preview: {text_preview}")

        chat_result = self._generate_wrapper.chat_template.parse_chat_response(
            accumulated_text
        )
        content_preview = chat_result.content[:100] if chat_result.content else None
        logger.debug(
            f"Parse result: content={content_preview}..., "
            f"tool_calls={chat_result.tool_calls}"
        )

        if chat_result.tool_calls:
            tool_calls = _convert_tool_calls(chat_result.tool_calls)
            logger.info(f"Found {len(tool_calls)} tool calls in stream")
            for i, tc in enumerate(tool_calls):
                logger.info(f"  Tool call {i}: {tc.function.name}({tc.function.arguments})")
            return tool_calls, "tool_calls"

        logger.info("No tool calls found in stream response")
        return None, "stop"

    def _create_stream_chunk(
        self,
        chat_id: str,
        model: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        finish_reason: Optional[str] = None,
        logprobs: Optional[Any] = None,
    ) -> ChatCompletionChunk:
        """Create a streaming chunk with the given content or tool calls.

        Args:
            chat_id: The chat completion ID
            model: The model name
            content: Text content for the delta (mutually exclusive with tool_calls)
            tool_calls: Tool calls for the delta (mutually exclusive with content)
            finish_reason: The finish reason (None for intermediate chunks)
            logprobs: Log probabilities if requested

        Returns:
            ChatCompletionChunk ready to yield
        """
        if tool_calls:
            delta = ChatMessage(role=Role.ASSISTANT, tool_calls=tool_calls)
        elif content:
            delta = ChatMessage(role=Role.ASSISTANT, content=content)
        else:
            delta = ChatMessage(role=Role.ASSISTANT)

        return ChatCompletionChunk(
            id=chat_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                    logprobs=logprobs,
                )
            ],
        )

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
                # Convert internal ToolCall format to OpenAI-compatible format
                openai_tool_calls = _convert_tool_calls(result.content.tool_calls)
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=final_content,
                    tool_calls=openai_tool_calls,
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
                            "tool_calls" if message.tool_calls else (result.finish_reason or "stop")
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
            logger.error(f"Failed to generate completion: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {e}") from e

    def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Stream generate OpenAI-compatible chunks with tool call support.

        Streams content chunks to the client while filtering out tool call XML.
        Tool calls are parsed from the accumulated text after streaming completes
        and emitted in a final chunk with finish_reason="tool_calls".

        For models that output tool calls in XML format (e.g., Qwen3-Coder),
        the XML is detected and filtered from the streamed output to avoid
        displaying raw XML to users. Duplicate tool calls are deduplicated.

        Args:
            request: The chat completion request containing messages and tools

        Yields:
            ChatCompletionChunk objects with streamed content and final tool calls
        """
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            params = self._prepare_generation_params(request)

            result = None
            accumulated_text = ""
            has_tools = request.tools is not None and len(request.tools) > 0
            in_tool_call = False
            pending_buffer = ""

            if has_tools:
                logger.info(f"Streaming with {len(request.tools)} tools available")

            # Stream content chunks
            for chunk in self._generate_wrapper.generate_stream(**params):
                # Extract content from chunk
                if chunk.content.text_delta:
                    content = chunk.content.text_delta
                    accumulated_text += content
                elif chunk.content.reasoning_delta:
                    content = chunk.content.reasoning_delta
                    accumulated_text += content
                else:
                    content = ""

                # Filter tool call XML when tools are available
                if has_tools and content:
                    stream_content, pending_buffer, in_tool_call = self._filter_stream_content(
                        content, pending_buffer, in_tool_call
                    )
                else:
                    stream_content = content

                # Yield content chunk
                if stream_content:
                    yield self._create_stream_chunk(
                        chat_id,
                        request.model,
                        content=stream_content,
                        logprobs=chunk.logprobs,
                    )
                result = chunk

            # Flush remaining buffer if not in tool call mode
            if pending_buffer and not in_tool_call and not _has_tool_call_marker(pending_buffer):
                yield self._create_stream_chunk(chat_id, request.model, content=pending_buffer)

            # Parse tool calls from accumulated text
            tool_calls = None
            finish_reason = "stop"
            if has_tools and accumulated_text:
                tool_calls, finish_reason = self._parse_stream_tool_calls(accumulated_text)

            # Emit final chunk with finish_reason and optional tool_calls
            yield self._create_stream_chunk(
                chat_id,
                request.model,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

            # Emit usage chunk if requested
            if (
                request.stream_options
                and request.stream_options.include_usage
                and result is not None
            ):
                cached_tokens = result.stats.cache_hit_tokens
                logger.debug(f"Stream response with {cached_tokens} cached tokens")

                prompt_tokens_details = None
                if cached_tokens > 0:
                    from .schema import PromptTokensDetails

                    prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT),
                            finish_reason=finish_reason,
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
