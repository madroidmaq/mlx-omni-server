import json
import re
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ....schemas.chat_schema import ChatMessage, Role
from ....schemas.tools_schema import ToolCall
from .chat_tokenizer import ChatTokenizer


class Llama3ChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "<|python_tag|>"
        self.end_tool_calls = "<|eom_id|>"

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse tool calls from model output.

        The model outputs function calls in the format:
        {"name": "function_name", "arguments": {"param1": "value1", ...}}
        """
        # Look for JSON patterns in the text
        tool_calls = []

        if text.startswith(self.start_tool_calls):
            # Match both old format with "parameters" and new format with "arguments"
            json_pattern = r'\{[^{}]*"name":\s*"[^"]+"\s*,\s*"(?:arguments|parameters)":\s*\{[^{}]+\}\}'
            matches = re.finditer(json_pattern, text)

            for match in matches:
                try:
                    # Parse the original format
                    tool_data = json.loads(match.group())

                    # Handle both "arguments" and "parameters" keys
                    params = tool_data.get("arguments", tool_data.get("parameters", {}))

                    # Create tool call using the helper method
                    tool_calls.append(
                        ToolCall.from_llama_output(
                            name=tool_data["name"],
                            parameters=params,
                            call_id=f"call_{uuid.uuid4().hex[:8]}",
                        )
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing tool call: {e}")
                    continue

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls if tool_calls else None,
        )
