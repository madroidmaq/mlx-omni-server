import json
import unittest
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.mlx.core_types import ToolCall
from mlx_omni_server.chat.mlx.tools.llama3 import Llama3ChatTokenizer


class TestLlama3ChatTokenizer(unittest.TestCase):
    def setUp(self):
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        self.tokenizer = Llama3ChatTokenizer(mock_tokenizer)
        self.invalid_responses = [
            """<?xml version="1.0" encoding="UTF-8"?>
            <json>
            {
                "name": "get_current_weather",
                "arguments": {"location": "Boston, MA", "unit": "celsius"}
            }
            </json>""",
            """```xml
            {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
            ```""",
            """<response>
            {
              "name": "get_current_weather",
              "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}
            }
            </response>""",
            """<|python_tag|>{"name": "get_current_weather", "parameters": {"location": "Boston", "unit": "celsius"}}<|eom_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            """{"type": "function", "name": "get_current_weather", "parameters": {"location": "Boston", "unit": "fahrenheit"}}<|eom_id|><|start_header_id|>assistant<|end_header_id|>

This JSON represents a function call to `get_current_weather` with the location set to "Boston" and the unit set to "fahrenheit".
            """,
        ]

    def test_strict_mode_decode_single_tool_call(self):
        # Test single tool call with double quotes
        self.tokenizer.strict_mode = True

        text = """<|python_tag|>{"name": "get_current_weather", "parameters": {"location": "Boston, MA", "unit": "fahrenheit"}}"""
        result = self.tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_current_weather")
        self.assertEqual(
            tool_call.arguments, {"location": "Boston, MA", "unit": "fahrenheit"}
        )

    def test_strict_mode_rejects_loose_format(self):
        # 确保严格模式下拒绝非标准格式
        self.tokenizer.strict_mode = True

        # Test with <response> tag (should fail in strict mode)
        text = """<response>
        {
          "name": "get_current_weather",
          "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}
        }
        </response>"""
        result = self.tokenizer.decode(text)

        self.assertIsNone(result)  # 应该返回 None

    def test_decode_invalid_tool_call(self):
        # Test invalid tool call format (missing name)
        text = """<tool_call>
    {"arguments": {"location": "Boston, MA"}}
    </tool_call>"""
        result = self.tokenizer.decode(text)

        self.assertIsNone(result)  # Should return None for invalid format

    def test_decode_invalid_json(self):
        # Test invalid JSON format

        for text in self.invalid_responses:
            result = self.tokenizer.decode(text)

            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)

            tool_call = result[0]
            self.assertEqual(tool_call.name, "get_current_weather")
            self.assertIsNotNone(tool_call.arguments)
