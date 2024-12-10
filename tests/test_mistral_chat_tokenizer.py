import json
import unittest
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.schemas.chat_schema import Role
from mlx_omni_server.services.chat.tools.llama3 import Llama3ChatTokenizer
from mlx_omni_server.services.chat.tools.mistral import MistralChatTokenizer


class TestMistralChatTokenizer(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = Mock(spec=TokenizerWrapper)
        self.llama_tokenizer = Llama3ChatTokenizer(self.mock_tokenizer)
        self.mistral_tokenizer = MistralChatTokenizer(self.mock_tokenizer)

    def test_mistral_decode_single_tool_call(self):
        # Test single tool call
        text = '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}]'
        result = self.mistral_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)

        tool_call = result.tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_current_weather")
        self.assertEqual(
            json.loads(tool_call.function.arguments), {"location": "Boston, MA"}
        )

    def test_mistral_decode_multiple_tool_calls(self):
        # Test multiple tool calls
        text = """[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}},
                               {"name": "get_forecast", "arguments": {"location": "New York, NY"}}]"""
        result = self.mistral_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 2)

        # Check first tool call
        tool_call1 = result.tool_calls[0]
        self.assertEqual(tool_call1.function.name, "get_current_weather")
        self.assertEqual(
            json.loads(tool_call1.function.arguments), {"location": "Boston, MA"}
        )

        # Check second tool call
        tool_call2 = result.tool_calls[1]
        self.assertEqual(tool_call2.function.name, "get_forecast")
        self.assertEqual(
            json.loads(tool_call2.function.arguments), {"location": "New York, NY"}
        )

    def test_mistral_decode_invalid_json(self):
        # Test invalid JSON format
        text = '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}'  # Missing closing bracket
        result = self.mistral_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(result.content, text)  # Should return original text
        self.assertIsNone(result.tool_calls)

    def test_mistral_decode_invalid_tool_call(self):
        # Test invalid tool call format (missing name)
        text = '[TOOL_CALLS] [{"arguments": {"location": "Boston, MA"}}]'
        result = self.mistral_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(
            result.content, text
        )  # Should return original text for invalid format
        self.assertIsNone(result.tool_calls)

    def test_mistral_decode_mixed_valid_invalid_calls(self):
        # Test mixture of valid and invalid tool calls
        text = """[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}},
                               {"arguments": {"location": "Invalid"}},
                               {"name": "get_forecast", "arguments": {"location": "New York, NY"}}]"""
        result = self.mistral_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 2)  # Should only have valid calls

        # Check valid tool calls
        self.assertEqual(result.tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(result.tool_calls[1].function.name, "get_forecast")

    def test_mistral_decode_non_tool_call(self):
        # Test regular text without tool call
        text = "This is a regular message"
        result = self.mistral_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(result.content, text)
        self.assertIsNone(result.tool_calls)


if __name__ == "__main__":
    unittest.main()
