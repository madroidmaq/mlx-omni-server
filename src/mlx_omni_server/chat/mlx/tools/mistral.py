import json
import uuid
from typing import List, Optional

from ..core_types import ToolCall
from ..core_types import ToolCall as CoreToolCall
from .base_tools import BaseToolParser


class MistralToolsParser(BaseToolParser):
    """Tools handler for Llama models."""

    def __init__(self):
        self.start_tool_calls = "[TOOL_CALLS]"
        self.end_tool_calls = ""

    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output.

        The model outputs function calls in the format:
        [TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}},
                     {"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}]

        Args:
            text: The model output text containing tool calls

        Returns:
            List of CoreToolCall objects or None if no tool calls found
        """
        # Look for JSON patterns in the text
        tool_calls = []

        if text.startswith(self.start_tool_calls):
            try:
                # Extract the JSON array from between square brackets after [TOOL_CALLS]
                json_str = text[len(self.start_tool_calls) :].strip()
                if json_str.startswith("[") and json_str.endswith("]"):
                    json_str = json_str.strip("[]").strip()
                    # Try to parse as array first
                    try:
                        tool_data = json.loads(f"[{json_str}]")
                    except json.JSONDecodeError:
                        # If array parsing fails, try single object
                        tool_data = json.loads(json_str)

                    # Handle both single object and array of objects
                    if isinstance(tool_data, dict):
                        tool_data = [tool_data]
                    elif not isinstance(tool_data, list):
                        raise ValueError(
                            "Invalid tool call format: expected dict or list"
                        )

                    for call in tool_data:
                        if not isinstance(call, dict) or "name" not in call:
                            continue

                        # Get arguments
                        args = call.get("arguments", call.get("parameters", {}))

                        # Create CoreToolCall object directly
                        tool_calls.append(
                            CoreToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                name=call["name"],
                                arguments=args,
                            )
                        )
                else:
                    # Invalid format, return None
                    return None
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing tool call: {e}")
                return None

        return tool_calls if tool_calls else None
