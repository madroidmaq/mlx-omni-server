import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mlx_lm.tokenizer_utils import TokenizerWrapper


class TokensDecoder(ABC):
    """Base class for tokens decoders."""

    @abstractmethod
    def decode(self, tokens: List[int]) -> Optional[Dict[str, Any]]:
        """Parse tool calls from model output."""
        pass

    def stream_decode(self, tokens: List[int]) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from model output."""
        return [self.decode(tokens)]


class ReasoningDecoder(TokensDecoder):
    """Base class for reasoning decoders."""

    thinking_tag: str = "think"
    enable_thinking: bool = False
    add_thinking_prefix: bool = False
    accumulated_text: str = ""

    def __init__(self, tokenizer: TokenizerWrapper):
        self.tokenizer = tokenizer
        self.accumulated_text = ""

    def set_thinking_prefix(self, add_thinking_prefix: bool) -> None:
        self.add_thinking_prefix = add_thinking_prefix
        self.accumulated_text = ""
        if add_thinking_prefix:
            self.accumulated_text = f"<{self.thinking_tag}>"

    def _parse_stream_response(self, text: str) -> Optional[Dict[str, Any]]:
        # Check if in thinking mode
        thinking_end_tag = f"</{self.thinking_tag}>"
        thinking_start_tag = f"<{self.thinking_tag}>"

        # Special case: text exactly equals the start tag
        if text == thinking_start_tag:
            self.accumulated_text += text
            return {"delta_content": None, "delta_reasoning": ""}

        # Special case: text exactly equals the end tag
        if text == thinking_end_tag:
            self.accumulated_text += text
            return {
                "delta_content": "",
                "delta_reasoning": None,
            }

        # Update accumulated text
        self.accumulated_text += text

        # Check if accumulated text already contains the end tag
        has_end_tag_before = thinking_end_tag in self.accumulated_text[: -len(text)]
        has_end_tag_now = thinking_end_tag in self.accumulated_text

        # If text starts with thinking tag and end tag hasn't been encountered yet
        if self.accumulated_text.startswith(thinking_start_tag) and not has_end_tag_now:
            # delta content as reasoning
            return {"delta_content": None, "delta_reasoning": text}
        # If current delta contains the end tag (from not having it to having it)
        elif not has_end_tag_before and has_end_tag_now:
            # Split the current delta
            parts = text.split(thinking_end_tag, 1)
            return {"delta_content": "", "delta_reasoning": None}
        # If end tag was already encountered before
        elif has_end_tag_before:
            # All content after the end tag is content
            return {"delta_content": text, "delta_reasoning": None}
        else:
            # Other cases, possibly thinking mode not enabled or other situations
            return {"delta_content": text, "delta_reasoning": None}

    def stream_decode(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tool calls from model output."""
        if not self.enable_thinking:
            return {"delta_content": text}

        return self._parse_stream_response(text)

    def _parse_response(self, response: str):
        tag = self.thinking_tag
        # First check for complete thinking tag pattern
        reasoning_regex = f"<{tag}>([\s\S]*?)</{tag}>"
        reasoning_match = re.search(reasoning_regex, response)

        if reasoning_match:
            # Extract thinking content
            reasoning_content = reasoning_match.group(1).strip()

            # Get final content by replacing thinking tag and its content
            content = re.sub(reasoning_regex, "", response, count=1).strip()

            return {
                "content": content,
                "reasoning": reasoning_content,
            }
        else:
            # Check if only end tag exists (missing start tag case)
            end_tag = f"</{tag}>"
            if end_tag in response:
                # Split response using end tag
                parts = response.split(end_tag, 1)
                if len(parts) == 2:
                    reasoning_content = parts[0].strip()
                    content = parts[1].strip()
                    return {
                        "content": content,
                        "reasoning": reasoning_content,
                    }

            # If no tags exist, the entire response is the content
            return {
                "content": response.strip(),
                "reasoning": None,
            }

    def decode(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse thinking content from model output"""
        if self.enable_thinking:
            return self._parse_response(text)

        return {"content": text}
