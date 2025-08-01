import unittest

from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser


class TestQwen3MoeToolParser(unittest.TestCase):
    def setUp(self):
        self.tools_parser = Qwen3MoeToolParser()

    def test_qwen3_moe_decode_single_tool_call(self):
        # Test single tool call as provided in the example
        text = """<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(
            tool_call.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

    def test_qwen3_moe_decode_multiple_tool_calls(self):
        # Test multiple tool calls as provided in the example
        text = """<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>
<tool_call>
<function=send_email>
<parameter=to>
john@example.com
</parameter>
<parameter=subject>
Meeting Tomorrow
</parameter>
<parameter=body>
Hi John, just confirming our meeting scheduled for tomorrow. Best regards!
</parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

        # Check first tool call
        tool_call1 = result[0]
        self.assertEqual(tool_call1.name, "get_weather")
        self.assertEqual(
            tool_call1.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

        # Check second tool call
        tool_call2 = result[1]
        self.assertEqual(tool_call2.name, "send_email")
        self.assertEqual(
            tool_call2.arguments,
            {
                "to": "john@example.com",
                "subject": "Meeting Tomorrow",
                "body": "Hi John, just confirming our meeting scheduled for tomorrow. Best regards!",
            },
        )

    def test_qwen3_moe_decode_tool_call_with_whitespace(self):
        # Test tool call with extra whitespace
        text = """  <tool_call>
  <function=get_weather>
  <parameter=location>
  San Francisco
  </parameter>
  <parameter=unit>
  fahrenheit
  </parameter>
  </function>
  </tool_call>  """
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(
            tool_call.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

    def test_qwen3_moe_decode_tool_call_no_parameters(self):
        # Test tool call without parameters
        text = """<tool_call>
<function=list_files>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "list_files")
        self.assertEqual(tool_call.arguments, {})

    def test_qwen3_moe_decode_invalid_tool_call_no_function(self):
        # Test invalid tool call format (missing function tag)
        text = """<tool_call>
<parameter=location>
San Francisco
</parameter>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNone(result)  # Should return None for invalid format

    def test_qwen3_moe_decode_malformed_xml(self):
        # Test malformed XML
        text = """<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
fahrenheit
</tool_call>"""  # Missing closing parameter tag
        result = self.tools_parser.parse_tools(text)

        # Should still work as we extract what we can
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[0].arguments, {"location": "San Francisco"})

    def test_qwen3_moe_decode_non_tool_call(self):
        # Test regular text without tool call
        text = "This is a regular message without any tool calls."
        result = self.tools_parser.parse_tools(text)

        self.assertIsNone(result)  # Should return None for non-tool call text

    def test_qwen3_moe_decode_mixed_content(self):
        # Test tool call mixed with regular text
        text = """Here's the weather information:

<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>

Let me check that for you."""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(
            tool_call.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

    def test_qwen3_moe_strict_mode(self):
        # Test strict mode behavior
        self.tools_parser.strict_mode = True

        # Valid format should work
        text = """<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)
        self.assertIsNotNone(result)

        # Mixed content should not work in strict mode
        text_mixed = """Some text before
<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
</function>
</tool_call>
Some text after"""
        result_mixed = self.tools_parser.parse_tools(text_mixed)
        self.assertIsNone(result_mixed)

    def test_qwen3_moe_decode_multiline_parameter(self):
        # Test parameter with multiline content
        text = """<tool_call>
<function=send_email>
<parameter=body>
Hello,

This is a multiline email body.
It contains multiple paragraphs.

Best regards,
Assistant
</parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "send_email")
        expected_body = """Hello,

This is a multiline email body.
It contains multiple paragraphs.

Best regards,
Assistant"""
        self.assertEqual(tool_call.arguments["body"], expected_body)

    def test_qwen3_moe_decode_missing_opening_tag(self):
        # Test case for missing opening <tool_call> tag (real-world scenario)
        text = """<function=get_weather>
<parameter=location>
San Francisco, CA
</parameter>
<parameter=unit>
celsius
</parameter>
</function>
</tool_call>
<tool_call>
<function=send_email>
<parameter=to>
john@example.com
</parameter>
<parameter=subject>
Meeting Tomorrow
</parameter>
<parameter=body>
Hi John, just confirming our meeting scheduled for tomorrow. Best regards!
</parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        # Should successfully parse both tool calls
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

        # First tool call (missing opening tag)
        first_call = result[0]
        self.assertEqual(first_call.name, "get_weather")
        self.assertEqual(
            first_call.arguments, {"location": "San Francisco, CA", "unit": "celsius"}
        )

        # Second tool call (complete format)
        second_call = result[1]
        self.assertEqual(second_call.name, "send_email")
        self.assertEqual(
            second_call.arguments,
            {
                "to": "john@example.com",
                "subject": "Meeting Tomorrow",
                "body": "Hi John, just confirming our meeting scheduled for tomorrow. Best regards!",
            },
        )


if __name__ == "__main__":
    unittest.main()
