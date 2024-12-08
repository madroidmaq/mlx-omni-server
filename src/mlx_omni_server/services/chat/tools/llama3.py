import json
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ....schemas.chat_schema import ChatMessage, Role
from ....schemas.tools_schema import FunctionCall, ToolCall
from ....utils.logger import logger
from .chat_tokenizer import ChatTokenizer


class Llama3ChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "<|python_tag|>"
        self.end_tool_calls = ""
        self.strict_mode = False
        self.pre_fill_tools_prompt = ""

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def _parse_strict_tools(self, text: str) -> Optional[List[ToolCall]]:
        tool_calls = []
        logger.debug(f"_parse_strict_tools: {text}")

        if text.strip().startswith(self.start_tool_calls):
            try:
                # Remove tool call tags and parse JSON directly
                json_str = text[len(self.start_tool_calls) :].strip()
                tool_data = json.loads(json_str)

                if isinstance(tool_data, dict) and "name" in tool_data:
                    # Get arguments and ensure they're a JSON string
                    args = tool_data.get("arguments", tool_data.get("parameters", {}))
                    if isinstance(args, str):
                        # Already a JSON string
                        arguments = args
                    else:
                        # Convert dict to JSON string
                        arguments = json.dumps(args)

                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            function=FunctionCall(
                                name=tool_data["name"],
                                arguments=arguments,
                            ),
                        )
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error parsing tool call: {e}")
                return None

        return tool_calls if tool_calls else None

    def _parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        """
        Parse tool calls from text using a simple JSON extraction method.
        Returns a list of ToolCall objects or None if no valid tool calls are found.
        """
        try:
            start = text.find("{")
            while start >= 0:
                count = 1
                pos = start + 1

                while pos < len(text) and count > 0:
                    if text[pos] == "{":
                        count += 1
                    elif text[pos] == "}":
                        count -= 1
                        if count == 0:
                            try:
                                json_str = text[start : pos + 1]
                                tool_data = json.loads(json_str)

                                if isinstance(tool_data, dict) and "name" in tool_data:
                                    params = tool_data.get(
                                        "arguments", tool_data.get("parameters", {})
                                    )
                                    if not isinstance(params, dict):
                                        params = {}

                                    return [
                                        ToolCall.from_llama_output(
                                            name=tool_data["name"],
                                            parameters=params,
                                            call_id=f"call_{uuid.uuid4().hex[:8]}",
                                        )
                                    ]
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.debug(f"Error parsing tool call: {e}")
                    pos += 1

                start = text.find("{", start + 1)

        except Exception as e:
            logger.error(f"Error during JSON extraction: {str(e)}")

        return None

    def decode(self, text: str) -> Optional[ChatMessage]:
        """
        Parse tool calls from model output.
        The model outputs function calls in JSON format with 'name' and optional 'arguments' fields.
        """
        response = self.pre_fill_tools_prompt + text
        tool_calls = None

        # 检查是否可能包含工具调用
        if self.strict_mode:
            tool_calls = self._parse_strict_tools(response)
        else:
            tool_calls = self._parse_tools(response)

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls,
        )
