import json
import unittest
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

# 更新导入路径指向 src 目录下的模块
from mlx_omni_server.schemas.chat_schema import Role
from mlx_omni_server.services.chat.tools.llama3 import Llama3ChatTokenizer


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

<|python_tag|>{"name": "get_current_weather", "parameters": {"location": "Boston", "unit": "fahrenheit"}}<|eom_id|><|start_header_id|>assistant<|end_header_id|>
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
        print(f"result: {result}")

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)

        tool_call = result.tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_current_weather")
        self.assertEqual(
            json.loads(tool_call.function.arguments),
            {"location": "Boston, MA", "unit": "fahrenheit"},
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

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(result.content, text)  # 应该返回原始文本
        self.assertIsNone(result.tool_calls)  # 不应该解析出工具调用

    def test_decode_invalid_tool_call(self):
        # Test invalid tool call format (missing name)
        text = """<tool_call>
    {"arguments": {"location": "Boston, MA"}}
    </tool_call>"""
        result = self.tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(
            result.content, text
        )  # Should return original text for invalid format
        self.assertIsNone(result.tool_calls)

    def test_decode_invalid_json(self):
        # Test invalid JSON format

        for text in self.invalid_responses:
            result = self.tokenizer.decode(text)

            self.assertIsNotNone(result)
            self.assertEqual(result.role, Role.ASSISTANT)
            self.assertIsNone(result.content)
            self.assertIsNotNone(result.tool_calls)
            self.assertEqual(len(result.tool_calls), 1)

            tool_call = result.tool_calls[0]
            self.assertEqual(tool_call.function.name, "get_current_weather")
            self.assertIsNotNone(tool_call.function.arguments)
