import json
import uuid
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.chat_schema import ChatMessage, Role
from mlx_omni_server.chat.tools_schema import (
    FunctionCall,
    SpecificToolChoice,
    Tool,
    ToolCall,
    ToolChoiceType,
)
from mlx_omni_server.utils.logger import logger

from .chat_tokenizer import ChatTokenizer
from .utils import parse_tool_calls


class Llama3ChatTokenizer(ChatTokenizer):
    """Tools handler for Llama models."""

    def __init__(self, tokenizer: TokenizerWrapper):
        super().__init__(tokenizer)
        self.start_tool_calls = "<|python_tag|>"
        self.end_tool_calls = ""
        self.strict_mode = False
        self.pre_fill_tools_prompt = ""

    def decode_stream(self, text: str, delta_text: str) -> Optional[List[ToolCall]]:
        pass

    def encode(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        **kwargs,
    ) -> str:
        prompt = super().encode(messages, tools, tool_choice, **kwargs)

        if tools:
            if isinstance(tool_choice, SpecificToolChoice):
                self.pre_fill_tools_prompt += self.start_tool_calls
                function_name = tool_choice.function["name"]

                self.pre_fill_tools_prompt += (
                    f"""{{"name": "{function_name}", "arguments":"""
                )

        return prompt + self.pre_fill_tools_prompt

    def _parse_strict_tools(self, text: str) -> Optional[List[ToolCall]]:
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
                logger.error(f"Error parsing tool call: {e}")
                return None

        return tool_calls if tool_calls else None

    def decode(self, text: str) -> Optional[ChatMessage]:
        """
        Parse tool calls from model output.
        The model outputs function calls in JSON format with 'name' and optional 'arguments' fields.
        """
        response = self.pre_fill_tools_prompt + text

        if self.strict_mode:
            tool_calls = self._parse_strict_tools(response)
        else:
            tool_calls = parse_tool_calls(response)

        return ChatMessage(
            role=Role.ASSISTANT,
            content=None if tool_calls else text,
            tool_calls=tool_calls,
        )