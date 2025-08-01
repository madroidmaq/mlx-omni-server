import json
import uuid
from typing import List, Optional

from mlx_omni_server.utils.logger import logger

from ..core_types import ToolCall
from ..core_types import ToolCall as CoreToolCall
from .base_tools import BaseToolParser


class Llama3ToolParser(BaseToolParser):
    """Tools handler for Llama models."""

    def __init__(self):
        self.start_tool_calls = "<|python_tag|>"
        self.end_tool_calls = ""
        self.strict_mode = False
        self.pre_fill_tools_prompt = ""

    def _parse_strict_tools(self, text: str) -> Optional[List[CoreToolCall]]:
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

                    # Create CoreToolCall object directly
                    tool_calls.append(
                        CoreToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=tool_data["name"],
                            arguments=args,
                        )
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error parsing tool call: {e}")
                return None

        return tool_calls if tool_calls else None

    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output.

        Args:
            text: Generated text that may contain tool calls

        Returns:
            List of platform-independent ToolCall objects or None if no tool calls found
        """
        response = self.pre_fill_tools_prompt + text

        if self.strict_mode:
            tool_calls = self._parse_strict_tools(response)
        else:
            # Use extract_tools for non-strict mode parsing
            from .base_tools import extract_tools

            tool_calls = extract_tools(response)

        return tool_calls
