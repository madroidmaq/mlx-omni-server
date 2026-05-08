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


class TestQwen3MoeToolParserJsonFormat(unittest.TestCase):
    """Tests for JSON-based tool call format (HuggingFace unified format)."""

    def setUp(self):
        self.tools_parser = Qwen3MoeToolParser()

    def test_qwen3_moe_decode_json_tool_call(self):
        # Qwen3 models can emit <tool_call>{"name":...,"arguments":...}</tool_call>
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "San Francisco"}}\n</tool_call>'
        result = self.tools_parser.parse_tools(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[0].arguments, {"location": "San Francisco"})

    def test_qwen3_moe_decode_multiple_json_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}\n</tool_call>'
        )
        result = self.tools_parser.parse_tools(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[1].name, "get_time")

    def test_qwen3_moe_json_fallback_to_xml(self):
        # XML format should still work when JSON parsing returns None
        text = """<tool_call>
<function=get_weather>
<parameter=location>
Boston
</parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[0].arguments, {"location": "Boston"})


class TestQwen3_5ToolParserIntegration(unittest.TestCase):
    """Integration tests for Qwen3.5 model (model_type='qwen3_5').

    Qwen3.5 models (e.g. mlx-community/Qwen3.5-0.8B-4bit) use the same XML
    tool call format as Qwen3-MoE, so they share the Qwen3MoeToolParser.
    These tests verify parsing of actual outputs observed from Qwen3.5 models.
    """

    def setUp(self):
        self.tools_parser = Qwen3MoeToolParser()
        # Set schema matching the get_weather tool used in real testing
        self.tools_parser.set_tools_schema(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]
        )

    def test_qwen3_5_single_tool_call(self):
        """Parse a single tool call as produced by Qwen3.5-0.8B-4bit."""
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>\n"
            "Sydney\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[0].arguments, {"location": "Sydney"})

    def test_qwen3_5_tool_call_with_preceding_text(self):
        """Qwen3.5 sometimes emits text before the tool call XML."""
        text = (
            "I will call the get_weather function for Sydney to check the "
            "current weather information. However, the function has the "
            "following parameters:\n\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>\n"
            "Sydney\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[0].arguments, {"location": "Sydney"})

    def test_qwen3_5_tool_call_with_trailing_text(self):
        """Qwen3.5 may emit text after the tool call XML in streaming."""
        text = (
            "I can analyze whether the user wants information from "
            "Tokyo's weather, as I am restricted to using "
            "`get_weather()` only.\n\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>\n"
            "Tokyo\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[0].arguments, {"location": "Tokyo"})

    def test_qwen3_5_multiple_tool_calls(self):
        """Parse multiple tool calls from Qwen3.5 output."""
        self.tools_parser.set_tools_schema(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "timezone": {"type": "string"},
                            },
                        },
                    },
                },
            ]
        )

        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>\n"
            "Sydney\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=get_time>\n"
            "<parameter=timezone>\n"
            "Australia/Sydney\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "get_weather")
        self.assertEqual(result[0].arguments, {"location": "Sydney"})
        self.assertEqual(result[1].name, "get_time")
        self.assertEqual(result[1].arguments, {"timezone": "Australia/Sydney"})

    def test_qwen3_5_tool_call_with_type_conversion(self):
        """Qwen3.5 XML parameters should be type-converted via schema."""
        self.tools_parser.set_tools_schema(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer"},
                                "include_meta": {"type": "boolean"},
                            },
                        },
                    },
                }
            ]
        )

        text = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>\n"
            "weather Sydney\n"
            "</parameter>\n"
            "<parameter=limit>\n"
            "5\n"
            "</parameter>\n"
            "<parameter=include_meta>\n"
            "true\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "search")
        self.assertEqual(result[0].arguments["query"], "weather Sydney")
        self.assertEqual(result[0].arguments["limit"], 5)
        self.assertIs(result[0].arguments["include_meta"], True)

    def test_qwen3_5_no_tool_call_in_plain_text(self):
        """Qwen3.5 plain text response should return None."""
        text = (
            "I don't have access to real-time weather data right this moment. "
            "Your next message will help me get the weather forecast today."
        )
        result = self.tools_parser.parse_tools(text)
        self.assertIsNone(result)


class TestQwen3_5RouterIntegration(unittest.TestCase):
    """Verify the full routing path from model_type to parser for Qwen3.5."""

    def test_qwen3_5_model_type_routes_to_xml_parser(self):
        """config['model_type'] == 'qwen3_5' must produce Qwen3MoeToolParser."""
        from mlx_omni_server.chat.mlx.tools.chat_template import load_tools_parser

        parser = load_tools_parser("qwen3_5")
        self.assertIsInstance(parser, Qwen3MoeToolParser)

    def test_qwen3_5_parser_has_correct_markers(self):
        """Qwen3.5 parser must use XML tool_call markers."""
        from mlx_omni_server.chat.mlx.tools.chat_template import load_tools_parser

        parser = load_tools_parser("qwen3_5")
        self.assertEqual(parser.start_tool_calls, "<tool_call>")
        self.assertEqual(parser.end_tool_calls, "</tool_call>")


if __name__ == "__main__":
    unittest.main()
