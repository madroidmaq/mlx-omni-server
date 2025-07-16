from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...schema import ToolChoice, ToolChoiceType
from ..core_types import ToolCall


class ChatTokenizer(ABC):
    """Base class for tools handlers."""

    start_tool_calls: str
    end_tool_calls: str

    def __init__(self, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer

    def encode(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
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

        if tools:
            if (
                isinstance(tool_choice, ToolChoice)
                and tool_choice == ToolChoice.REQUIRED
            ):
                prompt += self.start_tool_calls

        return prompt

    @abstractmethod
    def decode_stream(self, text: str) -> Optional[List[ToolCall]]:
        pass

    @abstractmethod
    def decode(self, text: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output.

        Args:
            text: Generated text that may contain tool calls

        Returns:
            List of platform-independent ToolCall objects or None if no tool calls found
        """
        pass
