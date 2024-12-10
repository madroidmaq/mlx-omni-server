import json
import re
import uuid
from typing import Optional

from mlx_omni_server.schemas.tools_schema import FunctionCall, ToolCall
from mlx_omni_server.utils.logger import logger


def _extract_tools(text: str):
    results = []

    pattern = (
        r'"name"\s*:\s*"([^"]+)"'  # Match name
        r"[^}]*?"  # Allow any characters in between
        r'(?:"arguments"|"parameters")'  # Match arguments or parameters
        r"\s*:\s*"  # Match colon and whitespace
        r"("  # Start capturing parameter value
        r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"  # Match nested objects
        r"|\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]"  # Match arrays
        r"|null"  # Match null
        r'|"[^"]*"'  # Match strings
        r")"  # End capturing
    )

    matches = re.finditer(pattern, text, re.DOTALL)

    matches_list = list(matches)
    for i, match in enumerate(matches_list):
        name, args_str = match.groups()
        # try:
        #     arguments = json.loads(args_str)
        #     print("Successfully parsed arguments as JSON")
        # except json.JSONDecodeError as e:
        #     print(f"Failed to parse JSON: {e}")
        #     arguments = args_str
        results.append({"name": name, "arguments": args_str})

    return results


def parse_tool_calls(text: str) -> Optional[list[dict]]:
    """
    Parse tool calls from text using regex to find JSON patterns containing name and arguments.
    Returns a list of ToolCall objects or None if no valid tool calls are found.
    """
    try:
        tool_calls = _extract_tools(text)
        if tool_calls:
            results = []
            for call in tool_calls:
                # Process arguments
                args = call["arguments"]
                arguments = args if isinstance(args, str) else json.dumps(args)

                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=arguments,
                    ),
                )
                results.append(tool_call)
            return results

        return None
    except Exception as e:
        logger.error(f"Error during regex matching: {str(e)}")
        return None
