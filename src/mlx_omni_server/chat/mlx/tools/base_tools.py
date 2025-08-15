import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from mlx_omni_server.utils.logger import logger

from ..core_types import ToolCall


def extract_tools(text: str) -> Optional[List[ToolCall]]:
    """Extract tool calls from text and return CoreToolCall objects directly.

    Args:
        text: Text containing tool calls

    Returns:
        List of CoreToolCall objects if tool calls are found, None otherwise
    """
    results = []

    pattern = (
        r'"name"\s*:\s*"([^"]+)"'  # Match name
        r"(?:"  # Start non-capturing group for optional arguments/parameters
        r"[^}]*?"  # Allow any characters in between
        r'(?:"arguments"|"parameters")'  # Match arguments or parameters
        r"\s*:\s*"  # Match colon and whitespace
        r"("  # Start capturing parameter value
        r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"  # Match nested objects
        r"|\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]"  # Match arrays
        r"|null"  # Match null
        r'|"[^"]*"'  # Match strings
        r")"  # End capturing
        r")?"  # Make the entire arguments/parameters section optional
    )

    matches = re.finditer(pattern, text, re.DOTALL)

    matches_list = list(matches)
    for i, match in enumerate(matches_list):
        name, args_str = match.groups()

        # Parse arguments from JSON string if provided
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments as JSON: {args_str}")
            arguments = {}

        # Create CoreToolCall object directly
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments
        )
        results.append(tool_call)

    return results if results else None


class BaseToolParser(ABC):
    start_tool_calls: str
    end_tool_calls: str

    @abstractmethod
    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        pass
