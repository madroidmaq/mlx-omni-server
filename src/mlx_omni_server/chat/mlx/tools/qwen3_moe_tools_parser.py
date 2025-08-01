import re
import uuid
from typing import List, Optional

from ....utils.logger import logger
from ..core_types import ToolCall
from .base_tools import BaseToolParser


class Qwen3MoeToolParser(BaseToolParser):
    """Simplified tools parser for Qwen3 MoE models.

    Parses tool calls in the XML-like format using regex patterns.
    Handles both complete and malformed tool call formats.
    """

    def __init__(self):
        self.start_tool_calls = "<tool_call>"
        self.end_tool_calls = "</tool_call>"
        self.strict_mode = False

    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output using simplified regex approach.

        Args:
            text: Generated text that may contain tool calls

        Returns:
            List of ToolCall objects or None if no tool calls found
        """
        if not text or not isinstance(text, str):
            return None

        try:
            # In strict mode, check format first
            if self.strict_mode and not self._is_strict_format(text):
                return None

            tool_calls = []

            # Pattern to match function name and parameters in XML format
            # Handles both <tool_call><function=name>...</function></tool_call>
            # and malformed <function=name>...</function></tool_call>
            pattern = r"<function=([^>]+)>(.*?)(?:</function>|</tool_call>)"

            matches = re.finditer(pattern, text, re.DOTALL)

            for match in matches:
                function_name = match.group(1).strip()
                function_content = match.group(2)

                if not function_name:
                    continue

                # Extract parameters from the function content
                arguments = self._extract_parameters(function_content)

                # Create ToolCall object
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=function_name,
                    arguments=arguments,
                )
                tool_calls.append(tool_call)

            return tool_calls if tool_calls else None

        except Exception as e:
            logger.error(f"Error parsing Qwen3 MoE tool calls: {e}")
            return None

    def _is_strict_format(self, text: str) -> bool:
        """Check if text follows strict tool call format."""
        stripped_text = text.strip()

        # In strict mode, text should start with <tool_call> and end with </tool_call>
        # and should not contain any text before or after the tool call
        if not (
            stripped_text.startswith(self.start_tool_calls)
            and stripped_text.endswith(self.end_tool_calls)
        ):
            return False

        # Should contain exactly one complete tool call
        tool_call_pattern = r"<tool_call>.*?</tool_call>"
        matches = re.findall(tool_call_pattern, text, re.DOTALL)
        return len(matches) == 1 and matches[0].strip() == stripped_text

    def _extract_parameters(self, content: str) -> dict:
        """Extract parameters from function content using regex.

        Args:
            content: Content inside <function>...</function> tags

        Returns:
            Dictionary of parameter name-value pairs
        """
        parameters = {}

        # Pattern to match <parameter=name>value</parameter>
        param_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"
        param_matches = re.finditer(param_pattern, content, re.DOTALL)

        for match in param_matches:
            param_name = match.group(1).strip()
            param_value = match.group(2).strip()

            if param_name:
                parameters[param_name] = param_value

        return parameters
