import json
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ....schemas.chat_schema import ChatMessage, Role
from ....schemas.tools_schema import FunctionCall, ToolCall
from .chat_tokenizer import ChatTokenizer


class Qwen2ChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "<tool_call>"
        self.end_tool_calls = "</tool_call>"
        self.strict_mode = False

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def _parse_strict_tools(self, text: str) -> Optional[list[dict]]:
        tool_calls = []

        if (
            text.strip().startswith(self.start_tool_calls)
            and self.end_tool_calls in text
        ):
            try:
                # Remove tool call tags and parse JSON directly
                json_str = text[
                    len(self.start_tool_calls) : text.find(self.end_tool_calls)
                ].strip()
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
                print(f"Error parsing tool call: {e}")
                return None

        return tool_calls

    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse tool calls from model output.

        The model outputs function calls in the format:
        <tool_call>
        {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
        </tool_call>

        Args:
            text: The model output text containing tool calls

        Returns:
            ChatMessage: A message containing the parsed tool calls
        """
        # Look for JSON patterns in the text
        tool_calls = []

        if self.strict_mode:
            tool_calls = self._parse_strict_tools(text)

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls if tool_calls else None,
        )
