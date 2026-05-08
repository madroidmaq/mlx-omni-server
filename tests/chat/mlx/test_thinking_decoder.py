import logging

import pytest

from mlx_omni_server.chat.mlx.core_types import ChatTemplateResult, StreamContent
from mlx_omni_server.chat.mlx.tools.thinking_decoder import ThinkingDecoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestThinkingDecoder:
    """Test functionality of the ThinkingDecoder class"""

    @pytest.fixture
    def decoder(self):
        """Create a ThinkingDecoder instance"""
        decoder = ThinkingDecoder()
        return decoder

    def test_parse_response_with_empty_thinking(self, decoder):
        """Test parsing responses with thinking tags"""
        # Prepare test data
        test_response = "<think>\n\n</think>\nHere is the final answer."

        # Execute test
        result = decoder._parse_response(test_response)

        # Verify results
        assert result["thinking"] == ""
        assert result["content"] == "Here is the final answer."

    def test_parse_response_with_thinking(self, decoder):
        """Test parsing responses with thinking tags"""
        # Prepare test data
        test_response = "<think>\nThis is a thinking process.\nAnalyzing the request.\n</think>\nHere is the final answer."

        # Execute test
        result = decoder._parse_response(test_response)

        # Verify results
        assert (
            result["thinking"] == "This is a thinking process.\nAnalyzing the request."
        )
        assert result["content"] == "Here is the final answer."

    def test_parse_response_without_thinking(self, decoder):
        """Test parsing responses without thinking tags"""
        # Prepare test data
        test_response = "This is a direct response without thinking tags."

        # Execute test
        result = decoder._parse_response(test_response)

        # Verify results
        assert result["thinking"] is None
        assert result["content"] == "This is a direct response without thinking tags."

    def test_decode_with_thinking_enabled(self, decoder):
        """Test decode method with thinking mode enabled"""
        # Prepare test data
        test_text = "<think>Reasoning process</think>Final answer"

        # Execute test
        result = decoder.decode(test_text)

        # Verify results
        assert result["thinking"] == "Reasoning process"
        assert result["content"] == "Final answer"

    def test_parse_stream_response_thinking_mode(self, decoder):
        """Test stream response parsing in thinking mode with sequential streaming calls"""
        # Reset state before test
        decoder.accumulated_text = ""

        # Step 1: Start with thinking tag
        result = decoder._parse_stream_response(f"<{decoder.thinking_tag}>")
        assert result["delta_content"] is None
        assert result["delta_thinking"] == ""

        # Step 2: First part of thinking content
        result = decoder._parse_stream_response("I'm thinking ")
        assert result["delta_content"] is None
        assert result["delta_thinking"] == "I'm thinking "

        # Step 3: Second part of thinking content
        result = decoder._parse_stream_response("about this problem.")
        assert result["delta_content"] is None
        assert result["delta_thinking"] == "about this problem."

        # Step 4: Receive end tag
        result = decoder._parse_stream_response(f"</{decoder.thinking_tag}>")
        assert result["delta_content"] == ""
        assert result["delta_thinking"] is None

        # Step 5: First part of final content
        result = decoder._parse_stream_response("Here is ")
        assert result["delta_content"] == "Here is "
        assert result["delta_thinking"] is None

        # Step 6: Second part of final content
        result = decoder._parse_stream_response("the answer.")
        assert result["delta_content"] == "the answer."
        assert result["delta_thinking"] is None

        # Verify accumulated text has everything
        expected_text = f"<{decoder.thinking_tag}>I'm thinking about this problem.</{decoder.thinking_tag}>Here is the answer."
        assert decoder.accumulated_text == expected_text

    def test_parse_response_missing_start_tag(self, decoder):
        """Test parsing responses with missing start tag but with end tag"""
        # Prepare test data that mimics the real-world example with missing start tag
        test_response = """Okay, the user is just greeting me.
</think>

Hello! How can I assist you today? 😊"""

        # Execute test
        result = decoder._parse_response(test_response)

        # Verify results
        expected_thinking = """Okay, the user is just greeting me."""
        expected_content = "Hello! How can I assist you today? 😊"

        assert result["thinking"] == expected_thinking
        assert result["content"] == expected_content


class TestStreamThinkingRouting:
    """Test that streaming thinking/content routing handles empty strings correctly.

    Regression tests for the bug where Python truthiness checks caused:
    - Bug 1: `if thinking:` treated empty string "" as falsy, misrouting to content
    - Bug 2: `content or response.text` fell back to raw token when content was ""
    """

    @pytest.fixture
    def decoder(self):
        return ThinkingDecoder()

    @staticmethod
    def _build_parse_result(decoder, text: str) -> ChatTemplateResult:
        """Simulate ChatTemplate.stream_parse_chat_result with a ThinkingDecoder."""
        result = decoder.stream_decode(text)
        delta_content = result.get("delta_content") or ""
        delta_thinking = result.get("delta_thinking")
        return ChatTemplateResult(content=delta_content, thinking=delta_thinking)

    @staticmethod
    def _route_to_stream_content(
        parse_result: ChatTemplateResult,
        raw_token: str,
        chunk_index: int = 0,
    ) -> StreamContent:
        """Replicate the routing logic from chat_generator.generate_stream()."""
        if parse_result.thinking is not None:
            return StreamContent(
                reasoning_delta=parse_result.thinking,
                token=0,
                chunk_index=chunk_index,
            )
        else:
            return StreamContent(
                text_delta=(
                    parse_result.content
                    if parse_result.content is not None
                    else raw_token
                ),
                token=0,
                chunk_index=chunk_index,
            )

    def test_open_think_tag_routes_to_reasoning(self, decoder):
        """Opening <think> tag yields thinking="" which must route to reasoning_delta."""
        parse_result = self._build_parse_result(decoder, "<think>")

        assert parse_result.thinking == ""
        assert parse_result.content == ""

        sc = self._route_to_stream_content(parse_result, raw_token="<think>")
        assert sc.reasoning_delta == ""
        assert sc.text_delta is None

    def test_thinking_tokens_route_to_reasoning(self, decoder):
        """Tokens inside <think> must route to reasoning_delta."""
        # First consume the opening tag
        decoder.stream_decode("<think>")

        parse_result = self._build_parse_result(decoder, "analyzing the problem")

        assert parse_result.thinking == "analyzing the problem"
        assert parse_result.content == ""

        sc = self._route_to_stream_content(
            parse_result, raw_token="analyzing the problem"
        )
        assert sc.reasoning_delta == "analyzing the problem"
        assert sc.text_delta is None

    def test_close_think_tag_routes_to_content(self, decoder):
        """Closing </think> tag yields thinking=None, routing to content branch."""
        decoder.stream_decode("<think>")
        decoder.stream_decode("some thought")

        parse_result = self._build_parse_result(decoder, "</think>")

        assert parse_result.thinking is None
        assert parse_result.content == ""

        sc = self._route_to_stream_content(parse_result, raw_token="</think>")
        # content is "" (not the raw </think> token)
        assert sc.text_delta == ""
        assert sc.reasoning_delta is None

    def test_content_after_thinking_uses_parsed_content(self, decoder):
        """Content tokens after </think> must use parsed content, not raw token."""
        decoder.stream_decode("<think>")
        decoder.stream_decode("thought")
        decoder.stream_decode("</think>")

        parse_result = self._build_parse_result(decoder, "Hello!")

        assert parse_result.thinking is None
        assert parse_result.content == "Hello!"

        sc = self._route_to_stream_content(parse_result, raw_token="Hello!")
        assert sc.text_delta == "Hello!"
        assert sc.reasoning_delta is None

    def test_no_thinking_decoder_uses_raw_token(self):
        """Without a ThinkingDecoder, content=text and thinking=None."""
        # Simulate ChatTemplate behavior when reason_decoder is None
        raw_token = "just a normal token"
        parse_result = ChatTemplateResult(content=raw_token, thinking=None)

        sc = self._route_to_stream_content(parse_result, raw_token=raw_token)
        assert sc.text_delta == raw_token
        assert sc.reasoning_delta is None

    def test_empty_content_no_fallback_to_raw_token(self, decoder):
        """When content is "" during thinking phase, must NOT fall back to raw token.

        This is the core regression test for Bug 2: `content or response.text`
        would return response.text when content was empty string.
        """
        decoder.stream_decode("<think>")

        # During thinking, a token like "analyzing" arrives
        raw_token = "analyzing"
        parse_result = self._build_parse_result(decoder, raw_token)

        # thinking is set, so routes to reasoning branch (content is irrelevant)
        assert parse_result.thinking is not None
        sc = self._route_to_stream_content(parse_result, raw_token=raw_token)
        assert sc.reasoning_delta == raw_token
        assert sc.text_delta is None

        # Now simulate the end tag transition where content=""
        decoder.stream_decode(raw_token)  # accumulate the thinking token
        parse_result = self._build_parse_result(decoder, "</think>")
        assert parse_result.content == ""
        assert parse_result.thinking is None

        # The key assertion: content="" should NOT be replaced by raw_token
        sc = self._route_to_stream_content(parse_result, raw_token="</think>")
        assert sc.text_delta == ""
        assert sc.text_delta != "</think>"

    def test_full_streaming_sequence(self, decoder):
        """End-to-end test: full streaming sequence produces correct routing."""
        tokens = ["<think>", "Let me ", "think.", "</think>", "\n\n", "Answer!"]
        expected = [
            ("reasoning", ""),  # opening tag: empty reasoning delta
            ("reasoning", "Let me "),
            ("reasoning", "think."),
            ("content", ""),  # closing tag: empty content delta
            ("content", "\n\n"),
            ("content", "Answer!"),
        ]

        for token, (expected_type, expected_value) in zip(tokens, expected):
            pr = self._build_parse_result(decoder, token)
            sc = self._route_to_stream_content(pr, raw_token=token)

            if expected_type == "reasoning":
                assert sc.reasoning_delta == expected_value, (
                    f"Token {token!r}: expected reasoning_delta={expected_value!r}, "
                    f"got reasoning_delta={sc.reasoning_delta!r}"
                )
                assert sc.text_delta is None
            else:
                assert sc.text_delta == expected_value, (
                    f"Token {token!r}: expected text_delta={expected_value!r}, "
                    f"got text_delta={sc.text_delta!r}"
                )
                assert sc.reasoning_delta is None
