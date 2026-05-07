from mlx_lm import load

from mlx_omni_server.chat.mlx.tools.chat_template import ChatTemplate


class TestChatTemplate:
    thinking_model_id = "mlx-community/Qwen3-0.6B-4bit-DWQ"
    nonthinking_model_id = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    tools_model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def test_thinking_enabled(self):
        # Test explicitly enabling thinking
        model, tokenizer = load(self.thinking_model_id)
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "hello"}]
        prompt = chat_template.apply_chat_template(
            messages=messages,
            enable_thinking_parse=True,
        )
        print(prompt)
        assert prompt.endswith("<think>")
        assert chat_template.enable_thinking_parse is True
        assert chat_template.reason_decoder is not None

    def test_thinking_disabled(self):
        # Test explicitly disabling thinking - should do no modification
        model, tokenizer = load(self.thinking_model_id)
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "hello"}]
        prompt = chat_template.apply_chat_template(
            messages=messages,
            enable_thinking_parse=False,
        )
        print(prompt)
        # Should not modify prompt, no thinking processing
        assert not prompt.endswith("<think>")
        assert "<think>\n\n</think>\n\n" not in prompt
        assert chat_template.enable_thinking_parse is False
        assert chat_template.reason_decoder is None

    def test_thinking_auto_detect(self):
        # Test comprehensive None behavior: default value and auto-detection
        model, tokenizer = load(self.thinking_model_id)
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        # Test 1: Default value should be None
        assert chat_template.enable_thinking_parse is None
        assert chat_template.reason_decoder is None

        # Test 2: No auto-detection when prompt doesn't end with <think>
        messages = [{"role": "user", "content": "hello"}]
        prompt = chat_template.apply_chat_template(messages=messages)
        print("No auto-detection:", prompt)
        assert "<think>" not in prompt
        assert chat_template.enable_thinking_parse is None
        assert chat_template.reason_decoder is None

        # Test 3: Auto-detection when prompt ends with <think>
        model2, tokenizer2 = load("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        chat_template2 = ChatTemplate(tools_parser_type="hf", tokenizer=tokenizer2)

        prompt2 = chat_template2.apply_chat_template(messages=messages)
        print("Auto-detection:", prompt2)
        # This model's template should end with <think>, triggering auto-detection
        assert "<think>" in prompt2
        assert chat_template2.enable_thinking_parse is True
        assert chat_template2.reason_decoder is not None

    def test_multimodal_content(self):
        """Test handling of multimodal content (text + other types)"""
        model, tokenizer = load(self.nonthinking_model_id)
        chat_template = ChatTemplate(tools_parser_type="hf", tokenizer=tokenizer)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    # Image type not yet supported - will be filtered out
                    {"type": "image", "image_url": "data:image/jpeg;base64,..."},
                    {"type": "text", "text": "Please describe it."},
                ],
            }
        ]
        prompt = chat_template.apply_chat_template(messages=messages)
        print(prompt)
        assert "What's in this image?" in prompt
        assert "Please describe it." in prompt
        assert "data:image/jpeg" not in prompt  # Non-text content should be filtered

    def test_assistant_prefill(self):
        """Test prefill mode with assistant message"""
        model, tokenizer = load(self.nonthinking_model_id)
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How"},
        ]
        prompt = chat_template.apply_chat_template(
            messages=messages,
        )
        print(prompt)
        # Should use continue_final_message=True for prefill
        assert prompt.endswith("Hi there! How")

    def test_tools_basic(self):
        """Test basic tool integration"""
        model, tokenizer = load(self.tools_model_id)
        chat_template = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        prompt = chat_template.apply_chat_template(messages=messages, tools=tools)
        print(prompt)
        assert chat_template.has_tools is True
        assert prompt.find("get_weather") != -1
        # Note: Tool inclusion in prompt depends on model/tokenizer support
        # The important thing is that has_tools flag is set correctly

    def test_tools_choice_variations(self):
        """Test different tool choice options"""
        model, tokenizer = load(self.tools_model_id)
        chat_template = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Call a function"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "test_func", "description": "Test function"},
            }
        ]

        # Test required choice
        prompt_required = chat_template.apply_chat_template(
            messages=messages, tools=tools, tool_choice="required"
        )
        assert chat_template.has_tools is True
        assert prompt_required.strip().endswith(chat_template.start_tool_calls)

        # Test auto choice (should not add prefix)
        chat_template2 = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)
        prompt_auto = chat_template2.apply_chat_template(
            messages=messages, tools=tools, tool_choice="auto"
        )
        assert chat_template2.has_tools is True
        assert not prompt_auto.strip().endswith(chat_template2.start_tool_calls)

        # Test none choice (should not add prefix)
        chat_template3 = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)
        prompt_none = chat_template3.apply_chat_template(
            messages=messages, tools=tools, tool_choice="none"
        )
        assert chat_template3.has_tools is True
        assert not prompt_none.strip().endswith(chat_template3.start_tool_calls)

    def test_thinking_with_tools(self):
        """Test thinking mode combined with tools"""
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit-DWQ")
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Use tools to help me"}]
        tools = [{"type": "function", "function": {"name": "helper"}}]

        prompt = chat_template.apply_chat_template(
            messages=messages, tools=tools, enable_thinking_parse=True
        )

        assert chat_template.has_tools is True
        assert chat_template.enable_thinking_parse is True
        assert prompt.endswith("<think>")

    def test_conversation_history(self):
        """Test multiple message conversation"""
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit-DWQ")
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why don't scientists trust atoms?"},
            {"role": "user", "content": "Why?"},
        ]

        prompt = chat_template.apply_chat_template(messages=messages)

        # All messages should be present in some form
        assert "Hi" in prompt or "hello" in prompt.lower()
        assert "joke" in prompt
        assert "atoms" in prompt

    def test_kwargs_passthrough(self):
        """Test that additional kwargs are passed through to tokenizer"""
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit-DWQ")
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "test"}]

        # This should not raise an error even with extra kwargs
        prompt = chat_template.apply_chat_template(
            messages=messages, custom_param="test_value"
        )
        assert isinstance(prompt, str)

    def test_tool_calls_json_conversion(self):
        """Test that tool_calls JSON arguments are converted to dicts for Jinja template."""
        from mlx_omni_server.chat.mlx.tools.chat_template import ChatTemplate

        # Test messages with OpenAI format tool_calls (JSON string arguments)
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC", "unit": "celsius"}',
                        },
                    }
                ],
            },
        ]

        # Mock tokenizer that verifies JSON arguments were converted to dicts
        class VerifyingMockTokenizer:
            def apply_chat_template(self, conversation, **_kwargs):
                for msg in conversation:
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            if "function" in tc and isinstance(tc["function"], dict):
                                func = tc["function"]
                                if "arguments" in func:
                                    if isinstance(func["arguments"], str):
                                        raise TypeError("Expected dict arguments")
                                    assert isinstance(func["arguments"], dict)
                                    assert func["arguments"]["location"] == "NYC"
                                    assert func["arguments"]["unit"] == "celsius"
                return "conversion successful"

        chat_template = ChatTemplate(
            tools_parser_type="qwen3", tokenizer=VerifyingMockTokenizer()
        )

        result = chat_template.apply_chat_template(messages=messages)
        assert result == "conversion successful"

    def test_openai_adapter_tool_call_conversion(self):
        """Test that _convert_tool_calls function converts internal format to OpenAI format"""
        import json

        from mlx_omni_server.chat.mlx.core_types import ToolCall as CoreToolCall
        from mlx_omni_server.chat.openai.openai_adapter import _convert_tool_calls
        from mlx_omni_server.chat.openai.schema import ToolCall as SchemaToolCall

        # Test with internal CoreToolCall format (dict arguments)
        core_tool_calls = [
            CoreToolCall(
                id="call_123",
                name="get_weather",
                arguments={"location": "NYC", "unit": "celsius"},  # dict format
            )
        ]

        # Convert to OpenAI format
        openai_tool_calls = _convert_tool_calls(core_tool_calls)

        # Verify conversion
        assert openai_tool_calls is not None
        assert len(openai_tool_calls) == 1

        tool_call = openai_tool_calls[0]
        assert isinstance(tool_call, SchemaToolCall)
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"

        # Arguments should be JSON string
        assert isinstance(tool_call.function.arguments, str)
        args_dict = json.loads(tool_call.function.arguments)
        assert args_dict == {"location": "NYC", "unit": "celsius"}

        print("✓ OpenAI adapter tool call conversion test passed")

    def test_tool_calls_none_content_coerced(self):
        """Test that assistant messages with tool_calls and content=None don't crash."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny, 25°C"},
            {"role": "user", "content": "Thanks"},
        ]

        class CapturingMockTokenizer:
            def apply_chat_template(self, conversation, **_kwargs):
                for msg in conversation:
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        assert (
                            msg["content"] == ""
                        ), f"Expected '', got {msg['content']!r}"
                return "ok"

        chat_template = ChatTemplate(
            tools_parser_type="qwen3", tokenizer=CapturingMockTokenizer()
        )
        result = chat_template.apply_chat_template(messages=messages)
        assert result == "ok"

    def test_openai_adapter_none_input(self):
        """Test that _convert_tool_calls handles None input correctly"""
        from mlx_omni_server.chat.openai.openai_adapter import _convert_tool_calls

        # Test with None input
        result = _convert_tool_calls(None)
        assert result is None

        # Test with empty list (should also return None due to "if not core_tool_calls")
        result = _convert_tool_calls([])
        assert result is None

        print("✓ OpenAI adapter None input test passed")
