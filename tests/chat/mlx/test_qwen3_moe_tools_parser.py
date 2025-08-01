import re
import unittest
from unittest.mock import patch

from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser


class TestQwen3MoeToolParser(unittest.TestCase):
    def setUp(self):
        self.tools_parser = Qwen3MoeToolParser()

    def test_qwen3_moe_decode_single_tool_call(self):
        """Test single tool call as provided in the example."""
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
        """Test multiple tool calls as provided in the example."""
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

    def test_qwen3_moe_decode_edge_cases(self):
        """Test various edge cases in a single parameterized test."""
        test_cases = [
            # Empty input
            ("", None, "empty string"),
            (None, None, "None input"),
            # Whitespace handling
            (
                """  <tool_call>
  <function=get_weather>
  <parameter=location>
  San Francisco
  </parameter>
  </function>
  </tool_call>  """,
                "get_weather",
                "extra whitespace",
            ),
            # Empty parameter values
            (
                """<tool_call>
<function=test>
<parameter=empty></parameter>
</function>
</tool_call>""",
                "test",
                "empty parameter",
            ),
            # Special characters in parameters
            (
                """<tool_call>
<function=test>
<parameter=special>Hello & <world> "quotes"</parameter>
</function>
</tool_call>""",
                "test",
                "special characters",
            ),
            # Unicode content
            (
                """<tool_call>
<function=translate>
<parameter=text>你好世界 🌍</parameter>
</function>
</tool_call>""",
                "translate",
                "unicode content",
            ),
        ]

        for i, (text, expected_name, description) in enumerate(test_cases):
            with self.subTest(case=i, description=description):
                if text is None:
                    with patch(
                        "mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser.logger"
                    ) as mock_logger:
                        result = self.tools_parser.parse_tools(text)
                        self.assertIsNone(result)
                        mock_logger.warning.assert_called_once()
                elif expected_name is None:
                    result = self.tools_parser.parse_tools(text)
                    self.assertIsNone(result)
                else:
                    result = self.tools_parser.parse_tools(text)
                    self.assertIsNotNone(result, f"Failed for case: {description}")
                    self.assertEqual(len(result), 1)
                    self.assertEqual(result[0].name, expected_name)

    def test_qwen3_moe_decode_tool_call_no_parameters(self):
        """Test tool call without parameters."""
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

    def test_qwen3_moe_decode_invalid_formats(self):
        """Test various invalid format scenarios."""
        invalid_cases = [
            # Missing function tag
            """<tool_call>
<parameter=location>
San Francisco
</parameter>
</tool_call>""",
            # Empty function name
            """<tool_call>
<function=>
<parameter=test>value</parameter>
</function>
</tool_call>""",
            # Function name with spaces only
            """<tool_call>
<function=   >
<parameter=test>value</parameter>
</function>
</tool_call>""",
        ]

        # These should return None due to missing/invalid function names
        for i, text in enumerate(invalid_cases):
            with self.subTest(case=i):
                with patch(
                    "mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser.logger"
                ) as mock_logger:
                    result = self.tools_parser.parse_tools(text)
                    self.assertIsNone(result)
                    # Verify appropriate warning was logged
                    self.assertTrue(mock_logger.warning.called)

        # This case should succeed but skip empty parameter name
        text_empty_param = """<tool_call>
<function=test>
<parameter=>value</parameter>
<parameter=valid>data</parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(text_empty_param)
        self.assertIsNotNone(result)
        # Empty parameter name should be skipped, only valid param should remain
        self.assertEqual(result[0].arguments, {"valid": "data"})

    def test_qwen3_moe_decode_malformed_xml_cases(self):
        """Test various malformed XML scenarios."""
        test_cases = [
            # Missing closing parameter tag - should still extract what's available
            (
                """<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
fahrenheit
</tool_call>""",
                "get_weather",
                {"location": "San Francisco"},
            ),
            # Missing function closing tag - should still work
            (
                """<tool_call>
<function=test>
<parameter=param>value</parameter>
</tool_call>""",
                "test",
                {"param": "value"},
            ),
        ]

        for i, (text, expected_name, expected_args) in enumerate(test_cases):
            with self.subTest(case=i):
                result = self.tools_parser.parse_tools(text)
                self.assertIsNotNone(result, f"Should parse malformed XML case {i}")
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].name, expected_name)
                self.assertEqual(result[0].arguments, expected_args)

        # Special case: Nested tool calls - regex doesn't match nested parameters properly
        nested_text = """<tool_call>
<function=outer>
<parameter=nested><tool_call><function=inner></function></tool_call></parameter>
</function>
</tool_call>"""
        result = self.tools_parser.parse_tools(nested_text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "outer")
        # The nested content breaks parameter parsing, so we get empty arguments
        self.assertEqual(result[0].arguments, {})

    def test_qwen3_moe_decode_non_tool_call(self):
        """Test regular text without tool call."""
        text = "This is a regular message without any tool calls."
        result = self.tools_parser.parse_tools(text)

        self.assertIsNone(result)  # Should return None for non-tool call text

    def test_qwen3_moe_decode_mixed_content(self):
        """Test tool call mixed with regular text."""
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

    def test_qwen3_moe_decode_complex_parameter_values(self):
        """Test complex parameter value processing."""
        test_cases = [
            # Multiline with proper indentation handling
            (
                """<tool_call>
<function=send_email>
<parameter=body>
    Hello,

    This is indented content.
    Multiple lines here.

    Best regards
</parameter>
</function>
</tool_call>""",
                "Hello,\n\nThis is indented content.\nMultiple lines here.\n\nBest regards",
            ),
            # Mixed indentation
            (
                """<tool_call>
<function=code>
<parameter=script>
  def hello():
      print("world")
      return True
</parameter>
</function>
</tool_call>""",
                'def hello():\n    print("world")\n    return True',
            ),
            # Empty lines preservation
            (
                """<tool_call>
<function=format>
<parameter=text>


Content


</parameter>
</function>
</tool_call>""",
                "Content",
            ),
        ]

        for i, (text, expected_value) in enumerate(test_cases):
            with self.subTest(case=i):
                result = self.tools_parser.parse_tools(text)
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 1)
                actual_value = list(result[0].arguments.values())[0]
                self.assertEqual(actual_value, expected_value)

    def test_qwen3_moe_strict_mode_comprehensive(self):
        """Test strict mode behavior comprehensively."""
        self.tools_parser.strict_mode = True

        valid_cases = [
            # Single tool call
            """<tool_call>
<function=test>
</function>
</tool_call>""",
            # With parameters
            """<tool_call>
<function=test>
<parameter=param>value</parameter>
</function>
</tool_call>""",
        ]

        invalid_cases = [
            # Mixed content
            """Text before
<tool_call>
<function=test>
</function>
</tool_call>""",
            # Multiple tool calls (not wrapped in single tag)
            """<tool_call>
<function=test1>
</function>
</tool_call>
<tool_call>
<function=test2>
</function>
</tool_call>""",
        ]

        # Test valid cases
        for i, text in enumerate(valid_cases):
            with self.subTest(valid_case=i):
                result = self.tools_parser.parse_tools(text)
                self.assertIsNotNone(
                    result, f"Valid case {i} should pass in strict mode"
                )

        # Test invalid cases
        for i, text in enumerate(invalid_cases):
            with self.subTest(invalid_case=i):
                result = self.tools_parser.parse_tools(text)
                self.assertIsNone(
                    result, f"Invalid case {i} should fail in strict mode"
                )

    def test_qwen3_moe_performance_and_error_handling(self):
        """Test performance with large inputs and error handling."""
        # Test with very large input (should not cause performance issues)
        large_text = (
            "Random text " * 10000
            + """<tool_call>
<function=test>
<parameter=data>content</parameter>
</function>
</tool_call>"""
        )

        result = self.tools_parser.parse_tools(large_text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "test")

        # Test with extremely nested content
        nested_text = (
            """<tool_call>
<function=nested>
<parameter=content>"""
            + "<nested>" * 100
            + "content"
            + "</nested>" * 100
            + """</parameter>
</function>
</tool_call>"""
        )

        result = self.tools_parser.parse_tools(nested_text)
        self.assertIsNotNone(result)

    def test_qwen3_moe_regex_error_handling(self):
        """Test regex compilation error handling."""
        # Test that regex errors are properly caught and handled
        with patch(
            "mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser.logger"
        ) as mock_logger:
            # Create text that would cause regex findall to be called
            test_text = "<tool_call><function=test></function></tool_call>"

            # Mock the _tool_call_pattern.findall method to raise error
            with patch.object(self.tools_parser, "_tool_call_pattern") as mock_pattern:
                mock_pattern.findall.side_effect = re.error("Test regex error")
                result = self.tools_parser.parse_tools(test_text)
                self.assertIsNone(result)
                mock_logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
