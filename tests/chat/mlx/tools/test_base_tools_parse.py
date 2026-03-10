import unittest

from mlx_omni_server.chat.mlx.tools.base_tools import extract_tools


class TestToolsParse(unittest.TestCase):
    examples = [
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
        """{"name": "get_random_fact_of_the_day", "{}"}""",
        """<|python_tag|>{"name": "analyze_health_data", "parameters": {"data": "[{"-""",
        """<tool_call>
        {"name": "generate_invoice", "arguments": {"transaction_details": {"product": "Laptop", "quantity": 2, "price": 1500}, "customer_details": {"name": "John Doe", "address": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}}}
        </tool_call>""",
        """[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}},{"name": "get_forecast", "arguments": {"location": "New York, NY"}}]""",
        '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA"}}]',
    ]

    def test_decode_tool_calls(self):
        # Test tool call extraction

        for text in self.examples:
            tools = extract_tools(text)

            self.assertIsNotNone(tools)
            print(f"tools: {tools}")

            tool_call = tools[0]
            self.assertIsNotNone(tool_call.name)
            self.assertIsNotNone(tool_call.arguments)

    def test_decode_deeply_nested_tool_call(self):
        # Deeply nested JSON inside arguments should not break parsing
        text = """<tool_call>
        {"name": "edit", "arguments": {"path": "w/test.file", "meta": {"a": {"b": {"c": {"d": 1}}}}}}
        </tool_call>"""
        tools = extract_tools(text)
        self.assertIsNotNone(tools)
        self.assertEqual(tools[0].name, "edit")
        self.assertIn("path", tools[0].arguments)
        self.assertEqual(tools[0].arguments["path"], "w/test.file")
        self.assertEqual(tools[0].arguments["meta"]["a"]["b"]["c"]["d"], 1)

    def test_decode_arguments_null(self):
        # "arguments": null should be treated as empty args, not a parse failure
        text = '<tool_call>\n{"name": "list_files", "arguments": null}\n</tool_call>'
        tools = extract_tools(text)
        self.assertIsNotNone(tools)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "list_files")
        self.assertEqual(tools[0].arguments, {})

    def test_malformed_tool_call_block_returns_none(self):
        # Malformed <tool_call> block should return None, not fall back to regex
        text = '<tool_call>\n{not valid json at all\n</tool_call>'
        tools = extract_tools(text)
        self.assertIsNone(tools)

    def test_decode_no_tool_calls(self):
        # Test when no tool calls are found
        text = "This is a regular message without any tool calls"
        tools = extract_tools(text)

        self.assertIsNone(tools)  # Should return None when no tool calls found
