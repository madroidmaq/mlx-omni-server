import json
import unittest
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_omni_server.chat.mlx.core_types import ToolCall
from mlx_omni_server.chat.mlx.tools.hugging_face import HuggingFaceChatTokenizer


class TestHuggingFaceChatTokenizer(unittest.TestCase):
    def setUp(self):
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        self.hf_tokenizer = HuggingFaceChatTokenizer(mock_tokenizer)

    def test_decode_single_tool_call(self):
        # Test single tool call with double quotes
        text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
</tool_call>"""
        result = self.hf_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_current_weather")
        self.assertEqual(
            tool_call.arguments,
            {"location": "Boston, MA", "unit": "fahrenheit"},
        )

    def test_decode_invalid_tool_call(self):
        # Test invalid tool call format (missing name)
        text = """<tool_call>
{"arguments": {"location": "Boston, MA"}}
</tool_call>"""
        result = self.hf_tokenizer.decode(text)

        self.assertIsNone(result)  # Should return None for invalid format

    def test_decode_non_tool_call(self):
        # Test non-tool call text
        text = "This is a regular message without any tool calls."
        result = self.hf_tokenizer.decode(text)

        self.assertIsNone(result)  # Should return None for non-tool call text

    def test_loose_mode_with_response_tag(self):
        # 启用宽松模式
        self.hf_tokenizer.strict_mode = False

        # Test with <response> tag
        text = """<response>
        {
          "name": "get_current_weather",
          "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}
        }
        </response>"""
        result = self.hf_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_current_weather")
        self.assertEqual(
            result[0].arguments,
            {"location": "Boston, MA", "unit": "fahrenheit"},
        )

    def test_loose_mode_with_function_tag(self):
        # 启用宽松模式
        self.hf_tokenizer.strict_mode = False

        # Test with <function-calls> tag
        text = """<function-calls>
          {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "celsius"}}
        </function-calls>"""
        result = self.hf_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_current_weather")
        self.assertEqual(
            result[0].arguments,
            {"location": "Boston, MA", "unit": "celsius"},
        )

    def test_loose_mode_with_markdown(self):
        # 启用宽松模式
        self.hf_tokenizer.strict_mode = False

        # Test with markdown code block
        text = """```xml
        {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
        ```"""
        result = self.hf_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_current_weather")

    def test_loose_mode_with_xml_declaration(self):
        # 启用宽松模式
        self.hf_tokenizer.strict_mode = False

        # Test with XML declaration
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <json>
        {
            "name": "get_current_weather",
            "arguments": {"location": "Boston, MA", "unit": "celsius"}
        }
        </json>"""
        result = self.hf_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_current_weather")
        self.assertEqual(
            result[0].arguments,
            {"location": "Boston, MA", "unit": "celsius"},
        )

    def test_strict_mode_rejects_loose_format(self):
        # 确保严格模式下拒绝非标准格式
        self.hf_tokenizer.strict_mode = True

        # Test with <response> tag (should fail in strict mode)
        text = """<response>
        {
          "name": "get_current_weather",
          "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}
        }
        </response>"""
        result = self.hf_tokenizer.decode(text)

        self.assertIsNone(result)  # 应该返回 None 表示没有解析出工具调用
