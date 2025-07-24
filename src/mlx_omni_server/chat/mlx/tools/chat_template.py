from abc import ABC
from typing import Any, Dict, List, Optional, Union

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ..core_types import ChatTemplateResult
from .base_tools import BaseToolParser
from .hugging_face import HuggingFaceToolParser
from .llama3 import Llama3ToolParser
from .mistral import MistralToolsParser
from .reasoning_decoder import ReasoningDecoder


def load_tools_parser(tools_parser_type: str) -> BaseToolParser:
    if tools_parser_type == "llama":
        return Llama3ToolParser()
    if tools_parser_type == "mistral":
        return MistralToolsParser()
    if tools_parser_type == "qwen2" or tools_parser_type == "qwen3":
        return HuggingFaceToolParser()
    else:
        return HuggingFaceToolParser()


class ChatTemplate(ABC):
    """Base class for tools handlers."""

    start_tool_calls: str
    end_tool_calls: str

    def __init__(self, tools_parser_type: str, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer
        self.has_tools = False
        self.reason_decoder = None
        self.enable_thinking: Optional[bool] = None
        self.tools_parser: Optional[BaseToolParser] = load_tools_parser(
            tools_parser_type
        )

        # Initialize tool call markers with default values
        self.start_tool_calls = self.tools_parser.start_tool_calls
        self.end_tool_calls = self.tools_parser.end_tool_calls

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """Encode tools and conversation into a prompt string.

        This is a common implementation that uses the tokenizer's chat template.
        Subclasses can override this if they need different behavior.
        """
        schema_tools = tools  # tools are already in dict format

        # Check if the last message is from assistant (for prefill)
        should_prefill = messages[-1].get("role") == "assistant"

        conversation = []
        for message in messages:
            # messages are already in dict format
            msg_dict = message.copy()  # Make a copy to avoid modifying original
            if isinstance(msg_dict.get("content"), list):
                msg_dict["content"] = "\n\n".join(
                    item["text"]
                    for item in msg_dict["content"]
                    if item.get("type") == "text"
                )
            conversation.append(msg_dict)

        if kwargs:
            self.enable_thinking = kwargs.pop("enable_thinking", None)

        if should_prefill:
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tools=schema_tools,
                enable_thinking=self.enable_thinking,
                tokenize=False,
                continue_final_message=True,
                **kwargs,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tools=schema_tools,
                enable_thinking=self.enable_thinking,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )

        if self.enable_thinking is True:
            self.reason_decoder = ReasoningDecoder(init_buffer="<think>")

            # Enable thinking: ensure prompt ends with <think> to trigger reasoning.
            if not prompt.rstrip().endswith("<think>"):
                prompt = prompt + "<think>"
        elif self.enable_thinking is False:
            # Disable thinking: immediately add a closed <think></think> block.
            if not prompt.rstrip().endswith("</think>"):
                prompt = prompt + "<think>\n\n</think>\n\n"
        else:
            # Initialize decoder if not yet done and prompt ends with <think>
            if prompt.rstrip().endswith("<think>"):
                self.enable_thinking = True
                self.reason_decoder = ReasoningDecoder(init_buffer="<think>")

        if tools:
            self.has_tools = True
            # Handle different tool_choice formats:
            # 1. String type: "auto", "required", "none"
            # 2. Dict type: {"type": "function", "function": {"name": "func_name"}} for forced specific function calls
            should_add_tool_calls = False

            if isinstance(tool_choice, str):
                should_add_tool_calls = tool_choice == "required"
            elif isinstance(tool_choice, dict):
                # TODO：The implementation logic needs further optimization.
                tool_type = tool_choice.get("type")
                should_add_tool_calls = (
                    tool_type == "function"
                )  # Only for forced specific function calls

            if should_add_tool_calls:
                prompt += self.start_tool_calls

        return prompt

    def stream_parse_chat_result(self, text: str) -> ChatTemplateResult:
        delta_content = text
        delta_reasoning = None

        if self.reason_decoder is not None:
            result = self.reason_decoder.stream_decode(text)
            delta_content = result.get("delta_content") or ""
            delta_reasoning = result.get("delta_reasoning")

        # TODO: support stream parse tools
        return ChatTemplateResult(
            content=delta_content,
            thinking=delta_reasoning,
        )

    def parse_chat_response(self, text: str) -> ChatTemplateResult:
        content = text
        thinking = None
        tool_calls = None

        if self.reason_decoder is not None:
            result = self.reason_decoder.decode(text)
            content = result.get("content")
            thinking = result.get("reasoning")

        if self.has_tools:
            tool_calls = self.tools_parser.parse_tools(content)

        return ChatTemplateResult(
            content=content, thinking=thinking, tool_calls=tool_calls
        )
