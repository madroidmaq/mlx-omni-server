from typing import List, Optional

from ....utils.logger import logger
from ..core_types import ToolCall
from .base_tools import BaseToolParser, extract_tools


class HuggingFaceToolParser(BaseToolParser):
    """Tools handler for Llama models.
    https://huggingface.co/blog/unified-tool-use
    """

    def __init__(self):
        self.start_tool_calls = "<tool_call>\n"
        self.end_tool_calls = "</tool_call>"
        self.strict_mode = False

    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output.

        Args:
            text: Generated text that may contain tool calls

        Returns:
            List of platform-independent ToolCall objects or None if no tool calls found
        """
        try:
            # In strict mode, only allow exact tool_call format
            if self.strict_mode:
                if not (
                    text.strip().startswith(self.start_tool_calls.strip())
                    and text.strip().endswith(self.end_tool_calls)
                ):
                    return None

            # Parse tool calls using extract_tools (now returns CoreToolCall objects directly)
            tool_calls = extract_tools(text)
            return tool_calls
        except Exception as e:
            logger.error(f"Error parsing tool calls: {e}")
            return None
