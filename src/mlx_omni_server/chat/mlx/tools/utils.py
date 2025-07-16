import json
import re
import uuid
from typing import List, Optional

from ....utils.logger import logger
from ..core_types import ToolCall as CoreToolCall


def extract_tools(text: str) -> Optional[List[CoreToolCall]]:
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
        tool_call = CoreToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments
        )
        results.append(tool_call)

    return results if results else None


# def parse_tool_calls(text: str) -> Optional[list[ToolCall]]:
#     """
#     Parse tool calls from text using regex to find JSON patterns containing name and arguments.
#     Returns a list of ToolCall objects or None if no valid tool calls are found.
#
#     Note: This function returns the original schema.ToolCall type for backward compatibility.
#     """
#     try:
#         core_tool_calls = extract_tools(text)
#         if core_tool_calls:
#             results = []
#             for core_call in core_tool_calls:
#                 # Convert CoreToolCall to original schema.ToolCall
#                 tool_call = ToolCall(
#                     id=core_call.id,
#                     function=FunctionCall(
#                         name=core_call.name,
#                         arguments=json.dumps(core_call.arguments),
#                     ),
#                 )
#                 results.append(tool_call)
#             return results
#
#         return None
#     except Exception as e:
#         logger.error(f"Error during regex matching: {str(e)}")
#         return None
