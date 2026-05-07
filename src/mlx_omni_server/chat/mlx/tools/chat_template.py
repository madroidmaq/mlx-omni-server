import json
import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Union

from mlx_lm.tokenizer_utils import TokenizerWrapper

logger = logging.getLogger(__name__)

from ..core_types import ChatTemplateResult
from .base_tools import BaseToolParser
from .glm45_tools_parser import GLM45ToolParser
from .hugging_face import HuggingFaceToolParser
from .llama3 import Llama3ToolParser
from .mistral import MistralToolsParser
from .qwen3_moe_tools_parser import Qwen3MoeToolParser
from .thinking_decoder import ThinkingDecoder

# Constants
THINK_TAG = "<think>"


def _parse_json_arg(args: Any) -> Any:
    """Parse JSON string argument to dict, return original if not valid JSON.

    Args:
        args: The argument value to parse (may be a JSON string or already parsed)

    Returns:
        Parsed dict/list if args was valid JSON string, otherwise original value
    """
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse tool call arguments as JSON: {e}")
            return args
    return args


def load_tools_parser(tools_parser_type: str) -> BaseToolParser:
    if tools_parser_type == "llama":
        return Llama3ToolParser()
    if tools_parser_type == "mistral":
        return MistralToolsParser()
    if tools_parser_type == "qwen2":
        return HuggingFaceToolParser()
    if tools_parser_type.startswith("qwen3"):
        return Qwen3MoeToolParser()
    if tools_parser_type == "glm4_moe":
        return GLM45ToolParser()
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
        self.enable_thinking_parse: Optional[bool] = None
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
            if msg_dict.get("content") is None:
                msg_dict["content"] = ""
            if isinstance(msg_dict.get("content"), list):
                msg_dict["content"] = "\n\n".join(
                    item["text"]
                    for item in msg_dict["content"]
                    if item.get("type") == "text"
                )
            # Convert tool_calls arguments from JSON string to dict for Jinja template
            # The Qwen3 chat template expects arguments as a dict, not a JSON string
            if msg_dict.get("tool_calls"):
                converted_tool_calls = []
                for tc in msg_dict["tool_calls"]:
                    # Convert Pydantic models to dict using model_dump()
                    if hasattr(tc, "model_dump"):
                        tc_dict = tc.model_dump()
                    elif isinstance(tc, dict):
                        tc_dict = tc.copy()
                    else:
                        # Unexpected type - add as-is without conversion
                        tc_dict = tc
                        converted_tool_calls.append(tc_dict)
                        continue

                    # Handle OpenAI format with nested function object
                    if "function" in tc_dict and isinstance(tc_dict["function"], dict):
                        func = tc_dict["function"]
                        if "arguments" in func:
                            func["arguments"] = _parse_json_arg(func["arguments"])
                    # Handle direct arguments field
                    elif "arguments" in tc_dict:
                        tc_dict["arguments"] = _parse_json_arg(tc_dict["arguments"])
                    converted_tool_calls.append(tc_dict)
                msg_dict["tool_calls"] = converted_tool_calls
            conversation.append(msg_dict)

        if kwargs:
            self.enable_thinking_parse = kwargs.pop("enable_thinking_parse", None)
            skip_thinking_prefill = kwargs.pop("skip_thinking_prefill", False)
        else:
            skip_thinking_prefill = False

        if should_prefill:
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tools=schema_tools,
                tokenize=False,
                continue_final_message=True,
                **kwargs,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tools=schema_tools,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )

        prompt = self._process_thinking_prompt(prompt, skip_thinking_prefill)

        if tools:
            self.has_tools = True
            # Pass tools schema to parser for type conversion
            if hasattr(self.tools_parser, "set_tools_schema"):
                self.tools_parser.set_tools_schema(tools)
            # Handle different tool_choice formats:
            # 1. String type: "auto", "required", "none"
            # 2. Dict type: {"type": "function", "function": {"name": "func_name"}}
            #    for forced specific function calls
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

    def _detect_thinking_from_prompt(self, prompt: str) -> bool:
        """Detect if prompt indicates thinking mode should be enabled.

        Args:
            prompt: The prompt string to analyze

        Returns:
            True if thinking should be auto-enabled, False otherwise
        """
        return prompt.rstrip().endswith(THINK_TAG)

    def _process_thinking_prompt(self, prompt: str, skip_thinking_prefill=False) -> str:
        """Process thinking-related prompt modifications.

        Logic overview:
        - enable_thinking_parse=None: Auto-detect if prompt ends with <think>, if so set to True
        - enable_thinking_parse=True: Extract thinking and content parts using current rules
        - enable_thinking_parse=False: No modification to prompt
        """
        enable_thinking_parse = self.enable_thinking_parse
        stripped_prompt = prompt.rstrip()  # Single rstrip call for efficiency

        # Auto-detect thinking if not explicitly set
        if enable_thinking_parse is None and self._detect_thinking_from_prompt(prompt):
            self.enable_thinking_parse = True
            enable_thinking_parse = True
        # If no <think> detected, remain None (no modification)

        if enable_thinking_parse is True:
            if skip_thinking_prefill:
                # With json_schema: ensure prompt doesn't end with <think>
                if stripped_prompt.endswith(THINK_TAG):
                    prompt = stripped_prompt[: -len(THINK_TAG)]
                # Let OutlinesLogitsProcessor handle thinking pattern
                self.reason_decoder = ThinkingDecoder(init_buffer=THINK_TAG)
            else:
                # Without json_schema: ensure prompt ends with <think>
                if not stripped_prompt.endswith(THINK_TAG):
                    prompt = prompt + THINK_TAG
                self.reason_decoder = ThinkingDecoder(init_buffer=THINK_TAG)

        elif enable_thinking_parse is False:
            # No modification to prompt
            self.reason_decoder = None

        # enable_thinking_parse is None: no modification, no decoder setup

        return prompt

    def stream_parse_chat_result(self, text: str) -> ChatTemplateResult:
        delta_content = text
        delta_thinking = None

        if self.reason_decoder is not None:
            result = self.reason_decoder.stream_decode(text)
            delta_content = result.get("delta_content") or ""
            delta_thinking = result.get("delta_thinking")

        # TODO: support stream parse tools
        return ChatTemplateResult(
            content=delta_content,
            thinking=delta_thinking,
        )

    def parse_chat_response(self, text: str) -> ChatTemplateResult:
        content = text
        thinking = None
        tool_calls = None

        if self.reason_decoder is not None:
            result = self.reason_decoder.decode(text)
            content = result.get("content")
            thinking = result.get("thinking")

        if self.has_tools:
            logger.debug(
                f"parse_chat_response: has_tools=True, parser type={type(self.tools_parser).__name__}"
            )
            tool_calls = self.tools_parser.parse_tools(content)

            # If tool calls were found, clear content to avoid duplication
            if tool_calls:
                logger.debug(
                    f"parse_chat_response: found {len(tool_calls)} tool call(s)"
                )
                content = ""
            else:
                logger.debug("parse_chat_response: no tool calls found by parser")
        else:
            logger.debug("parse_chat_response: has_tools=False, skipping tool parsing")

        return ChatTemplateResult(
            content=content, thinking=thinking, tool_calls=tool_calls
        )
