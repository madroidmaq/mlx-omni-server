from abc import ABC, abstractmethod
from typing import List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...schema import ChatMessage, Role, Tool, ToolCall, ToolChoice, ToolChoiceType


class ChatTokenizer(ABC):
    """Base class for tools handlers."""

    start_tool_calls: str
    end_tool_calls: str

    def __init__(self, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer

    def encode(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[ToolChoiceType] = None,
        **kwargs,
    ) -> str:
        """Encode tools and conversation into a prompt string.

        This is a common implementation that uses the tokenizer's chat template.
        Subclasses can override this if they need different behavior.
        """
        schema_tools = None
        if tools:
            schema_tools = [tool.model_dump(exclude_none=True) for tool in tools]

        should_prefill = messages[-1].role == Role.ASSISTANT

        conversation = []
        for message in messages:
            msg_dict = message.model_dump(exclude_none=True)
            if isinstance(msg_dict.get("content"), list):
                msg_dict["content"] = "\n\n".join(
                    item["text"]
                    for item in msg_dict["content"]
                    if item.get("type") == "text"
                )
            # Only set empty string for user/assistant roles that require content
            if msg_dict.get("content") is None and msg_dict["role"] in (
                Role.USER,
                Role.ASSISTANT,
            ):
                msg_dict["content"] = ""
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
        """Parse tool calls from model output."""
        pass

    @abstractmethod
    def decode(self, text: str) -> Optional[ChatMessage]:
        """Parse tool calls from model output."""
        pass
