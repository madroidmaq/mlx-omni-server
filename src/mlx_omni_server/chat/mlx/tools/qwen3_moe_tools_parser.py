import json
import re
import uuid
from typing import Any, Dict, List, Optional

from ....utils.logger import logger
from ..core_types import ToolCall
from .base_tools import BaseToolParser, extract_tools


class Qwen3MoeToolParser(BaseToolParser):
    """Simplified tools parser for Qwen3 MoE models.

    Parses tool calls in the XML-like format using regex patterns.
    Handles both complete and malformed tool call formats.

    Supports schema-based type conversion to match OpenAI API format:
    - integer parameters are converted to int
    - number parameters are converted to float
    - boolean parameters are converted to bool
    - null values are converted to None
    - array/object parameters are parsed from JSON
    """

    def __init__(self) -> None:
        self.start_tool_calls = "<tool_call>"
        self.end_tool_calls = "</tool_call>"
        self.strict_mode = False
        self._tools_schema: Dict[str, Dict[str, Any]] = {}

    def set_tools_schema(self, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        """Set tools schema for type conversion.

        Args:
            tools: List of tool definitions in OpenAI format
        """
        self._tools_schema = {}
        if tools:
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    name = func.get("name")
                    if name:
                        self._tools_schema[name] = func.get("parameters", {})

    def _get_param_type(self, func_name: str, param_name: str) -> str:
        """Get expected type for a parameter from schema.

        Args:
            func_name: Name of the function
            param_name: Name of the parameter

        Returns:
            Type string from schema, or "string" if not found
        """
        schema = self._tools_schema.get(func_name, {})
        properties = schema.get("properties", {})
        param_schema = properties.get(param_name, {})
        return param_schema.get("type", "string")

    def _convert_param_value(
        self, value: str, param_type: str, func_name: str, param_name: str
    ) -> Any:
        """Convert string value to appropriate type based on schema.

        Args:
            value: Raw string value from XML
            param_type: Expected type from schema
            func_name: Function name (for logging)
            param_name: Parameter name (for logging)

        Returns:
            Converted value with appropriate type
        """
        # Handle null for any type
        if value.lower() == "null":
            return None

        if param_type == "integer":
            try:
                return int(value)
            except ValueError:
                logger.debug(f"Failed to convert '{value}' to integer for {func_name}.{param_name}")
                return value

        if param_type == "number":
            try:
                return float(value)
            except ValueError:
                logger.debug(f"Failed to convert '{value}' to number for {func_name}.{param_name}")
                return value

        if param_type == "boolean":
            lower_value = value.lower()
            if lower_value == "true":
                return True
            if lower_value == "false":
                return False
            logger.debug(f"Unexpected boolean value '{value}' for {func_name}.{param_name}")
            return value

        if param_type in ("array", "object"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON for {func_name}.{param_name}: {value}")
                return value

        # Default: return as string
        return value

    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output using simplified regex approach.

        Args:
            text: Generated text that may contain tool calls

        Returns:
            List of ToolCall objects or None if no tool calls found
        """
        if not text or not isinstance(text, str):
            logger.debug("parse_tools: empty or non-string input")
            return None

        # Log if schema is available
        logger.debug(
            f"parse_tools: schema available for functions: {list(self._tools_schema.keys())}"
        )

        try:
            # In strict mode, check format first
            if self.strict_mode and not self._is_strict_format(text):
                logger.debug("parse_tools: text doesn't match strict format")
                return None

            # First: prefer <tool_call>{"name":...,"arguments":...}</tool_call> JSON blocks.
            # Qwen3 models frequently emit the HuggingFace unified format, not <function=...> tags.
            parsed = extract_tools(text)
            if parsed:
                logger.debug(f"parse_tools: extracted {len(parsed)} tool call(s) via base_tools.extract_tools")
                return parsed

            tool_calls: List[ToolCall] = []

            # Fallback: match <function=name>...</function> style tool calls
            # Handles both <tool_call><function=name>...</function></tool_call>
            # and malformed <function=name>...</function></tool_call>
            pattern = r"<function=([^>]+)>(.*?)(?:</function>|</tool_call>)"

            # Check if text contains tool_call markers
            if "<tool_call>" in text or "<function=" in text:
                logger.debug("parse_tools: detected potential tool call markers in text")
            else:
                logger.debug("parse_tools: no tool call markers found")

            matches = re.finditer(pattern, text, re.DOTALL)

            for match in matches:
                function_name = match.group(1).strip()
                function_content = match.group(2)
                logger.debug(f"parse_tools: found function match - name='{function_name}'")

                if not function_name:
                    continue

                # Extract parameters from the function content
                arguments = self._extract_parameters(function_content, function_name)
                logger.debug(f"parse_tools: extracted arguments for {function_name}: {arguments}")

                # Create ToolCall object
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=function_name,
                    arguments=arguments,
                )
                tool_calls.append(tool_call)

            if tool_calls:
                logger.debug(f"parse_tools: successfully parsed {len(tool_calls)} tool call(s)")
            else:
                logger.debug("parse_tools: no tool calls extracted from text")

        except Exception as e:
            logger.error(f"Error parsing Qwen3 MoE tool calls: {e}")
            return None
        else:
            return tool_calls or None

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

    def _extract_parameters(self, content: str, function_name: str) -> Dict[str, Any]:
        """Extract parameters from function content using regex.

        Args:
            content: Content inside <function>...</function> tags
            function_name: Name of the function (for type lookup)

        Returns:
            Dictionary of parameter name-value pairs with appropriate types
        """
        parameters: Dict[str, Any] = {}

        # Pattern to match <parameter=name>value</parameter>
        param_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"
        param_matches = re.finditer(param_pattern, content, re.DOTALL)

        for match in param_matches:
            param_name = match.group(1).strip()
            param_value = match.group(2).strip()

            if param_name:
                # Warn if parameter name already exists (duplicate)
                if param_name in parameters:
                    logger.warning(
                        f"Duplicate parameter '{param_name}' in {function_name}, "
                        "last value will be used"
                    )
                # Get expected type from schema and convert
                param_type = self._get_param_type(function_name, param_name)
                converted_value = self._convert_param_value(
                    param_value, param_type, function_name, param_name
                )
                parameters[param_name] = converted_value

        return parameters
