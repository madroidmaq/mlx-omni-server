"""Anthropic Messages API Adapter

This module provides an adapter to convert between Anthropic Messages API format
and the internal MLX generation interface.
"""

import uuid
from typing import Any, Dict, Generator, List, Optional

from mlx_omni_server.chat.anthropic.anthropic_schema import (
    AnthropicTool,
    ContentBlock,
    InputMessage,
    MessagesRequest,
    MessagesResponse,
    MessageStreamEvent,
    RequestTextBlock,
    RequestToolResultBlock,
    RequestToolUseBlock,
    StopReason,
    StreamDelta,
    StreamEventType,
    SystemPrompt,
    TextBlock,
    ThinkingBlock,
    ThinkingConfigEnabled,
    ToolUseBlock,
    Usage,
)
from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator
from mlx_omni_server.utils.logger import logger


class AnthropicMessagesAdapter:
    """Anthropic Messages API adapter with internal parameter management."""

    def __init__(self, wrapper: ChatGenerator):
        """Initialize adapter with wrapper object.

        Args:
            wrapper: ChatGenerator instance (cached and ready to use)
        """
        self._default_max_tokens = 2048
        self._generate_wrapper = wrapper

    def _convert_system_to_messages(
        self, system: Optional[SystemPrompt], messages: List[InputMessage]
    ) -> List[Dict[str, Any]]:
        """Convert system prompt and messages to MLX format.

        Args:
            system: System prompt (string or list of text blocks)
            messages: Input messages

        Returns:
            List of messages in MLX format
        """
        mlx_messages = []

        # Convert system prompt to system message if present
        if system:
            system_content = ""
            if isinstance(system, str):
                system_content = system
            else:
                # List of SystemTextBlock
                system_content = "\n".join(block.text for block in system)

            mlx_messages.append(
                {
                    "role": "system",
                    "content": system_content,
                }
            )

        # Convert input messages
        for msg in messages:
            mlx_msg = {
                "role": msg.role.value,
            }

            # Handle content
            if isinstance(msg.content, str):
                mlx_msg["content"] = msg.content
            else:
                # List of content blocks - convert to appropriate format
                content_parts = []
                for block in msg.content:
                    if isinstance(block, RequestTextBlock):
                        content_parts.append(block.text)
                    elif isinstance(block, RequestToolUseBlock):
                        # Handle tool use blocks
                        mlx_msg["tool_calls"] = [
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": block.input,
                                },
                            }
                        ]
                    elif isinstance(block, RequestToolResultBlock):
                        # Handle tool result blocks
                        tool_content = block.content
                        if isinstance(tool_content, str):
                            content_parts.append(tool_content)
                        else:
                            # List of blocks
                            for sub_block in tool_content:
                                if isinstance(sub_block, RequestTextBlock):
                                    content_parts.append(sub_block.text)

                        mlx_msg["tool_call_id"] = block.tool_use_id
                        if block.is_error:
                            mlx_msg["name"] = "error"
                    # Note: Image blocks would be handled here too

                if content_parts:
                    mlx_msg["content"] = "\n".join(content_parts)

            mlx_messages.append(mlx_msg)

        return mlx_messages

    def _convert_tools_to_mlx(
        self, tools: Optional[List[AnthropicTool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Anthropic tools to MLX format.

        Args:
            tools: List of Anthropic tools

        Returns:
            List of tools in MLX format
        """
        if not tools:
            return None

        mlx_tools = []
        for tool in tools:
            mlx_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": tool.input_schema.type,
                        "properties": tool.input_schema.properties or {},
                        "required": tool.input_schema.required or [],
                    },
                },
            }
            mlx_tools.append(mlx_tool)

        return mlx_tools

    def _prepare_generation_params(self, request: MessagesRequest) -> Dict[str, Any]:
        """Prepare parameters for MLX generation.

        Args:
            request: Anthropic Messages API request

        Returns:
            Parameters for ChatGenerator
        """
        # Convert messages
        messages = self._convert_system_to_messages(request.system, request.messages)

        # Convert tools
        tools = self._convert_tools_to_mlx(request.tools)

        # Template parameters
        template_kwargs = {}

        # Handle thinking configuration
        if request.thinking and isinstance(request.thinking, ThinkingConfigEnabled):
            template_kwargs["enable_thinking"] = True
            # template_kwargs["thinking_budget"] = request.thinking.budget_tokens

        # Prepare sampler configuration
        sampler_config = {
            "temp": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
            "top_k": request.top_k or 0,
        }

        logger.info(f"Anthropic messages: {messages}")
        logger.info(f"Anthropic template_kwargs: {template_kwargs}")

        params = {
            "messages": messages,
            "tools": tools,
            "max_tokens": request.max_tokens,
            "sampler": sampler_config,
            "template_kwargs": template_kwargs,
            "enable_prompt_cache": True,
        }

        # Note: ChatGenerator doesn't currently support stop_sequences
        # This is a known limitation that will be addressed in the future
        # if request.stop_sequences:
        #     params["stop_sequences"] = request.stop_sequences

        return params

    def _create_content_blocks(
        self,
        text_content: Optional[str],
        reasoning_content: Optional[str],
        tool_calls: Optional[List[Any]] = None,
    ) -> List[ContentBlock]:
        """Create content blocks from generation result.

        Args:
            text_content: Main text content
            reasoning_content: Thinking/reasoning content
            tool_calls: Tool calls from generation

        Returns:
            List of content blocks
        """
        blocks = []

        # Add thinking block first if present
        if reasoning_content:
            blocks.append(ThinkingBlock(thinking=reasoning_content))

        # Add text block if present
        if text_content:
            blocks.append(TextBlock(text=text_content))

        # Add tool use blocks
        if tool_calls:
            for tool_call in tool_calls:
                blocks.append(
                    ToolUseBlock(
                        id=tool_call.id,
                        name=tool_call.name,
                        input=tool_call.arguments,
                    )
                )

        # Ensure we always have at least one content block
        if not blocks:
            blocks.append(TextBlock(text=""))

        return blocks

    def _map_finish_reason(
        self, finish_reason: Optional[str], has_tool_calls: bool
    ) -> StopReason:
        """Map internal finish reason to Anthropic format.

        Args:
            finish_reason: Internal finish reason
            has_tool_calls: Whether response has tool calls

        Returns:
            Anthropic stop reason
        """
        if has_tool_calls:
            return StopReason.TOOL_USE

        if finish_reason == "stop":
            return StopReason.END_TURN
        elif finish_reason == "length":
            return StopReason.MAX_TOKENS
        elif finish_reason == "stop_sequence":
            return StopReason.STOP_SEQUENCE
        else:
            return StopReason.END_TURN

    def generate(self, request: MessagesRequest) -> MessagesResponse:
        """Generate complete response using the wrapper.

        Args:
            request: Anthropic Messages API request

        Returns:
            Anthropic Messages API response
        """
        try:
            # Prepare parameters
            params = self._prepare_generation_params(request)

            # Generate using wrapper
            result = self._generate_wrapper.generate(**params)

            # Create content blocks
            content_blocks = self._create_content_blocks(
                text_content=result.content.text,
                reasoning_content=result.content.reasoning,
                tool_calls=result.content.tool_calls,
            )

            # Map stop reason
            stop_reason = self._map_finish_reason(
                result.finish_reason, bool(result.content.tool_calls)
            )

            # Create usage statistics
            cached_tokens = result.stats.cache_hit_tokens
            usage = Usage(
                input_tokens=result.stats.prompt_tokens + cached_tokens,
                output_tokens=result.stats.completion_tokens,
                cache_read_input_tokens=cached_tokens if cached_tokens > 0 else None,
            )

            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                content=content_blocks,
                model=request.model,
                stop_reason=stop_reason,
                usage=usage,
            )

        except Exception as e:
            logger.error(
                f"Failed to generate Anthropic completion: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    def generate_stream(
        self, request: MessagesRequest
    ) -> Generator[MessageStreamEvent, None, None]:
        """Generate streaming response.

        Args:
            request: Anthropic Messages API request

        Yields:
            Anthropic streaming events
        """
        try:
            message_id = f"msg_{uuid.uuid4().hex[:24]}"

            # Prepare parameters
            params = self._prepare_generation_params(request)

            # Start message event
            yield MessageStreamEvent(
                type=StreamEventType.MESSAGE_START,
                message=MessagesResponse(
                    id=message_id,
                    content=[],
                    model=request.model,
                    stop_reason=None,
                    usage=Usage(input_tokens=0, output_tokens=0),
                ),
            )

            # Track content for final message
            accumulated_text = ""
            accumulated_reasoning = ""
            final_result = None
            current_block_index = 0
            in_thinking = False

            for chunk in self._generate_wrapper.generate_stream(**params):
                # Determine content type and send appropriate events
                if chunk.content.reasoning_delta:
                    # Thinking content
                    if not in_thinking:
                        # Start thinking block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=ThinkingBlock(thinking=""),
                        )
                        in_thinking = True

                    # Thinking delta
                    yield MessageStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=current_block_index,
                        delta=StreamDelta(
                            type="thinking_delta",
                            thinking=chunk.content.reasoning_delta,
                        ),
                    )
                    accumulated_reasoning += chunk.content.reasoning_delta

                elif chunk.content.text_delta:
                    # Text content
                    if in_thinking:
                        # End thinking block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_STOP,
                            index=current_block_index,
                        )
                        current_block_index += 1
                        in_thinking = False

                        # Start text block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=TextBlock(text=""),
                        )
                    elif not accumulated_text:
                        # First text chunk - start text block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=TextBlock(text=""),
                        )

                    # Text delta
                    yield MessageStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=current_block_index,
                        delta=StreamDelta(
                            type="text_delta", text=chunk.content.text_delta
                        ),
                    )
                    accumulated_text += chunk.content.text_delta

                final_result = chunk

            # Add signature delta for thinking blocks if we had thinking content
            if in_thinking and accumulated_reasoning:
                # Add a placeholder signature for thinking block integrity
                yield MessageStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_DELTA,
                    index=current_block_index,
                    delta=StreamDelta(
                        type="signature_delta", signature="placeholder_signature_hash"
                    ),
                )

            # End final content block
            yield MessageStreamEvent(
                type=StreamEventType.CONTENT_BLOCK_STOP, index=current_block_index
            )

            # Map stop reason and usage
            if final_result:
                cached_tokens = final_result.stats.cache_hit_tokens
                usage = Usage(
                    input_tokens=final_result.stats.prompt_tokens + cached_tokens,
                    output_tokens=final_result.stats.completion_tokens,
                    cache_read_input_tokens=(
                        cached_tokens if cached_tokens > 0 else None
                    ),
                )

                stop_reason = self._map_finish_reason(
                    final_result.finish_reason,
                    False,  # StreamContent doesn't have tool_calls, so always False
                )
            else:
                usage = Usage(input_tokens=0, output_tokens=0)
                stop_reason = StopReason.END_TURN

            # Message delta event with stop reason and usage
            yield MessageStreamEvent(
                type=StreamEventType.MESSAGE_DELTA,
                delta=StreamDelta(stop_reason=stop_reason),
                usage=usage,
            )

            # Message stop event (no delta)
            yield MessageStreamEvent(type=StreamEventType.MESSAGE_STOP)

        except Exception as e:
            logger.error(
                f"Error during Anthropic stream generation: {str(e)}", exc_info=True
            )
            raise
