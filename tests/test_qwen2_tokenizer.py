import json
import unittest
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

# 更新导入路径指向 src 目录下的模块
from mlx_omni_server.schemas.chat_schema import Role
from mlx_omni_server.services.chat.tools.qwen2 import Qwen2ChatTokenizer


class TestQwen2ChatTokenizer(unittest.TestCase):
    def setUp(self):
        mock_tokenizer = Mock(spec=TokenizerWrapper)
        self.qwen2_tokenizer = Qwen2ChatTokenizer(mock_tokenizer)

    def test_decode_single_tool_call(self):
        # Test single tool call with double quotes
        text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
</tool_call>"""
        result = self.qwen2_tokenizer.decode(text)

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

    def test_decode_invalid_json(self):
        # Test invalid JSON format
        text = """<tool_call>
{"name": "get_current_weather", invalid_json}
</tool_call>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(
            result.content, text
        )  # Should return original text for invalid JSON
        self.assertIsNone(result.tool_calls)

    def test_decode_invalid_tool_call(self):
        # Test invalid tool call format (missing name)
        text = """<tool_call>
{"arguments": {"location": "Boston, MA"}}
</tool_call>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(
            result.content, text
        )  # Should return original text for invalid format
        self.assertIsNone(result.tool_calls)

    def test_decode_non_tool_call(self):
        # Test non-tool call text
        text = "This is a regular message without any tool calls."
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(result.content, text)
        self.assertIsNone(result.tool_calls)

    def test_loose_mode_with_response_tag(self):
        # 启用宽松模式
        self.qwen2_tokenizer.strict_mode = False

        # Test with <response> tag
        text = """<response>
        {
          "name": "get_current_weather",
          "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}
        }
        </response>"""

        #         text = """<function-calls>
        #   {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "celsius"}}
        # </function-calls>
        #         """
        result = self.qwen2_tokenizer.decode(text)
        print(f"=======\nresult: \n{result}")

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {"location": "Boston, MA", "unit": "fahrenheit"},
        )

    def test_loose_mode_with_function_tag(self):
        # 启用宽松模式
        self.qwen2_tokenizer.strict_mode = False

        # Test with <function-calls> tag
        text = """<function-calls>
          {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "celsius"}}
        </function-calls>"""
        result = self.qwen2_tokenizer.decode(text)
        print(f"=======\nresult: \n{result}")

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertIsNone(result.content)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {"location": "Boston, MA", "unit": "celsius"},
        )

    def test_loose_mode_with_markdown(self):
        # 启用宽松模式
        self.qwen2_tokenizer.strict_mode = False

        # Test with markdown code block
        text = """```xml
        {"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}
        ```"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_current_weather")

    def test_loose_mode_with_xml_declaration(self):
        # 启用宽松模式
        self.qwen2_tokenizer.strict_mode = False

        # Test with XML declaration
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <json>
        {
            "name": "get_current_weather",
            "arguments": {"location": "Boston, MA", "unit": "celsius"}
        }
        </json>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {"location": "Boston, MA", "unit": "celsius"},
        )

    def test_loose_mode_with_reversed_fields(self):
        # 启用宽松模式
        self.qwen2_tokenizer.strict_mode = False

        # Test with reversed fields order (arguments before name)
        text = """<tools>
        {
            "arguments": {"location": "Boston", "unit": "celsius"},
            "name": "get_current_weather"
        }
        </tools>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {"location": "Boston", "unit": "celsius"},
        )

    def test_strict_mode_rejects_loose_format(self):
        # 确保严格模式下拒绝非标准格式
        self.qwen2_tokenizer.strict_mode = True

        # Test with <response> tag (should fail in strict mode)
        text = """<response>
        {
          "name": "get_current_weather",
          "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}
        }
        </response>"""
        result = self.qwen2_tokenizer.decode(text)

        self.assertIsNotNone(result)
        self.assertEqual(result.role, Role.ASSISTANT)
        self.assertEqual(result.content, text)  # 应该返回原始文本
        self.assertIsNone(result.tool_calls)  # 不应该解析出工具调用
