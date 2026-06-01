"""Anthropic Messages API Adapter

This module provides an adapter to convert between Anthropic Messages API format
and the internal MLX generation interface.
"""

import json
import os
import re
import uuid
from typing import Any, Dict, Generator, List, Optional

from mlx_omni_server.chat.anthropic.anthropic_schema import (
    RequestThinkingBlock,
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
        # Claude Code 2.1.154+ injects role=system into messages[]; hoist to system block.
        for msg in messages:
            if msg.role.value == "system":
                extra = msg.content if isinstance(msg.content, str) else " ".join(
                    b.text for b in msg.content if hasattr(b, "text")
                )
                if extra:
                    existing = mlx_messages[0]["content"] if mlx_messages and mlx_messages[0]["role"] == "system" else None
                    if existing is not None:
                        mlx_messages[0]["content"] = existing + "\n\n" + extra
                    else:
                        mlx_messages.insert(0, {"role": "system", "content": extra})
                continue
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
                        # Strip tool call XML that leaked into text content
                        clean = re.sub(r"<tool_call>.*?</tool_call>", "", block.text, flags=re.DOTALL).strip()
                        clean = re.sub(r"<function=.*?</tool_call>", "", clean, flags=re.DOTALL).strip()
                        if clean:
                            content_parts.append(clean)
                    elif isinstance(block, RequestToolUseBlock):
                        # Handle tool use blocks
                        logger.debug(f"tool_use block: id={block.id} name={block.name} input={block.input}")
                        mlx_msg["tool_calls"] = [
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
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
                    elif isinstance(block, RequestThinkingBlock):
                        pass  # Thinking blocks are not passed to the model
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
            "temp": request.temperature or float(os.environ.get("MLX_OMNI_TEMPERATURE", 1.0)),
            "top_p": request.top_p or float(os.environ.get("MLX_OMNI_TOP_P", 1.0)),
            "top_k": request.top_k or int(os.environ.get("MLX_OMNI_TOP_K", 0)),
            **({"min_p": getattr(request, "min_p", None) if getattr(request, "min_p", None) is not None else float(os.environ.get("MLX_OMNI_MIN_P") or 0)} if (getattr(request, "min_p", None) is not None or os.environ.get("MLX_OMNI_MIN_P")) else {}),
            **({"repetition_penalty": getattr(request, "repetition_penalty", None) or float(os.environ.get("MLX_OMNI_REPETITION_PENALTY") or 1.0)} if (getattr(request, "repetition_penalty", None) is not None or os.environ.get("MLX_OMNI_REPETITION_PENALTY")) else {}),
        }

        logger.info(f"Anthropic messages: {messages}")
        logger.info(f"Anthropic template_kwargs: {template_kwargs}")

        params = {
            "messages": messages,
            "tools": tools,
            "max_tokens": int(os.environ.get("MLX_OMNI_MAX_TOKENS") or request.max_tokens),
            **({"max_kv_size": int(os.environ.get("MLX_OMNI_CONTEXT_SIZE"))} if os.environ.get("MLX_OMNI_CONTEXT_SIZE") else {}),
            "sampler": sampler_config,
            "template_kwargs": template_kwargs,
            "enable_prompt_cache": True,
        }

        # Note: ChatGenerator doesn't currently support stop_sequences
        # This is a known limitation that will be addressed in the future
        # if request.stop_sequences:
        #     params["stop_sequences"] = request.stop_sequences

        # Stop words from env var (mlx-lm does not support stop sequences natively)
        stop_words = []
        env_stop = os.environ.get("MLX_OMNI_STOP_WORDS")
        if env_stop:
            stop_words = [w for w in env_stop.split(",") if w.strip()]
        if request.stop_sequences:
            stop_words.extend(request.stop_sequences)
        params["_stop_words"] = stop_words
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
        if has_tool_calls or finish_reason == "tool_calls":
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
            _stop_words = params.pop("_stop_words", [])
            result = self._generate_wrapper.generate(**params)
            # Apply stop words post-hoc for non-streaming
            if _stop_words and result.content.text:
                for sw in _stop_words:
                    idx = result.content.text.find(sw)
                    if idx != -1:
                        result.content.text = result.content.text[:idx]
                        break

            # Create content blocks
            logger.debug(f"generate result text: {result.content.text[:300]!r}")
            logger.debug(f"generate result tool_calls: {result.content.tool_calls}")
            logger.debug(f"generate finish_reason: {result.finish_reason}")
            # Strip tool call XML from text so it does not leak into the text block
            clean_text = re.sub(r"<tool_call>.*?</tool_call>", "", result.content.text, flags=re.DOTALL).strip()
            clean_text = re.sub(r"<function=.*?</tool_call>", "", clean_text, flags=re.DOTALL).strip()
            content_blocks = self._create_content_blocks(
                text_content=clean_text or None,
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

            accumulated_text = []
            in_text_block = False
            _stop_words = params.pop("_stop_words", [])
            _stop_hit = False
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
                    # Smart streaming: emit text until tool_call tag starts, then buffer
                    if in_thinking:
                        # End thinking block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_STOP,
                            index=current_block_index,
                        )
                        current_block_index += 1
                        in_thinking = False
                    accumulated_text += chunk.content.text_delta
                    # Check stop words
                    if _stop_words:
                        full_so_far_sw = "".join(accumulated_text)
                        for sw in _stop_words:
                            sw_idx = full_so_far_sw.find(sw)
                            if sw_idx != -1:
                                # Truncate accumulated text at stop word
                                accumulated_text = list(full_so_far_sw[:sw_idx])
                                _stop_hit = True
                                break
                    if _stop_hit:
                        break
                    full_so_far = "".join(accumulated_text)
                    # Suppress streaming once we see a tool_call opening tag
                    tool_call_start = full_so_far.find("<tool_call>")
                    if tool_call_start == -1:
                        # No tool call yet - stream this chunk normally
                        if not in_text_block:
                            yield MessageStreamEvent(
                                type=StreamEventType.CONTENT_BLOCK_START,
                                index=current_block_index,
                                content_block=TextBlock(text=""),
                            )
                            in_text_block = True
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_DELTA,
                            index=current_block_index,
                            delta=StreamDelta(
                                type="text_delta", text=chunk.content.text_delta
                            ),
                        )
                    elif tool_call_start > 0 and not in_text_block:
                        # Tool call starts mid-chunk - emit text before it
                        pre_text = full_so_far[:tool_call_start]
                        if pre_text.strip():
                            yield MessageStreamEvent(
                                type=StreamEventType.CONTENT_BLOCK_START,
                                index=current_block_index,
                                content_block=TextBlock(text=""),
                            )
                            in_text_block = True
                            yield MessageStreamEvent(
                                type=StreamEventType.CONTENT_BLOCK_DELTA,
                                index=current_block_index,
                                delta=StreamDelta(type="text_delta", text=pre_text),
                            )
                    # else: tool call already started, just accumulate


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


            # Map stop reason and usage
            # Parse tool calls from accumulated text
            full_text = "".join(accumulated_text)
            logger.debug(f"stream full text: {repr(full_text)[:500]}")
            parsed_tool_calls = None
            if hasattr(self._generate_wrapper, "chat_template") and self._generate_wrapper.chat_template and hasattr(self._generate_wrapper.chat_template, "tools_parser"):
                parser = self._generate_wrapper.chat_template.tools_parser
                if parser:
                    parsed_tool_calls = parser.parse_tools(full_text)
            logger.debug(f"stream parsed tool_calls: {parsed_tool_calls}")

            # Emit buffered text block only if there is clean non-tool-call text
            clean_text = re.sub(r"<tool_call>.*?</tool_call>", "", full_text, flags=re.DOTALL).strip()
            clean_text = re.sub(r"<function=.*?</tool_call>", "", clean_text, flags=re.DOTALL).strip()
            if clean_text and not parsed_tool_calls:
                yield MessageStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_START,
                    index=current_block_index,
                    content_block=TextBlock(text=""),
                )
                yield MessageStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_DELTA,
                    index=current_block_index,
                    delta=StreamDelta(type="text_delta", text=clean_text),
                )
                yield MessageStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_STOP,
                    index=current_block_index,
                )
                current_block_index += 1

            # Emit tool use blocks after text block is closed
            if parsed_tool_calls:
                for tool_call in parsed_tool_calls:
                    current_block_index += 1
                    yield MessageStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_START,
                        index=current_block_index,
                        content_block=ToolUseBlock(
                            id=tool_call.id,
                            name=tool_call.name,
                            input={},
                        ),
                    )
                    yield MessageStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=current_block_index,
                        delta=StreamDelta(
                            type="input_json_delta",
                            partial_json=json.dumps(tool_call.arguments),
                        ),
                    )
                    yield MessageStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_STOP,
                        index=current_block_index,
                    )

            if final_result:
                cached_tokens = final_result.stats.cache_hit_tokens
                usage = Usage(
                    input_tokens=final_result.stats.prompt_tokens + cached_tokens,
                    output_tokens=final_result.stats.completion_tokens,
                    cache_read_input_tokens=(
                        cached_tokens if cached_tokens > 0 else None
                    ),
                )
                logger.debug(f"finish_reason={final_result.finish_reason} parsed_tool_calls={parsed_tool_calls}")
                stop_reason = self._map_finish_reason(
                    final_result.finish_reason,
                    bool(parsed_tool_calls),
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
