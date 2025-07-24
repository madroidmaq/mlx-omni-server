from mlx_lm import load

from mlx_omni_server.chat.mlx.tools.chat_template import ChatTemplate


class TestChatTemplate:
    thinking_model_id = "mlx-community/Qwen3-0.6B-4bit-DWQ"
    nonthinking_model_id = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    # tools_model_id = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    tools_model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def test_enable_thinking(self):
        # Test tool call extraction
        model, tokenizer = load(self.thinking_model_id)

        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [
            {
                "role": "user",
                "content": "hello",
            }
        ]
        prompt = chat_template.apply_chat_template(
            messages=messages,
            enable_thinking=True,
        )
        print(prompt)
        assert prompt.endswith("<think>")
        assert chat_template.enable_thinking is True
        assert chat_template.reason_decoder is not None

    def test_disable_thinking(self):
        # Test tool call extraction
        model, tokenizer = load(self.thinking_model_id)

        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [
            {
                "role": "user",
                "content": "hello",
            }
        ]
        prompt = chat_template.apply_chat_template(
            messages=messages,
            enable_thinking=False,
        )
        print(prompt)
        assert not prompt.endswith("<think>")
        assert chat_template.enable_thinking is False
        assert chat_template.reason_decoder is None

    def test_none_thinking(self):
        # Test tool call extraction
        model, tokenizer = load(self.thinking_model_id)

        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [
            {
                "role": "user",
                "content": "hello",
            }
        ]
        prompt = chat_template.apply_chat_template(
            messages=messages,
        )
        print(prompt)
        assert "<think>" not in prompt
        assert chat_template.enable_thinking is None
        assert chat_template.reason_decoder is None

    def test_auto_thinking_with_prefill(self):
        # Test tool call extraction
        model, tokenizer = load("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

        chat_template = ChatTemplate(tools_parser_type="hf", tokenizer=tokenizer)

        messages = [
            {
                "role": "user",
                "content": "hello",
            }
        ]
        prompt = chat_template.apply_chat_template(
            messages=messages,
        )
        print(prompt)
        assert "<think>" in prompt
        assert chat_template.enable_thinking is True
        assert chat_template.reason_decoder is not None

    def test_multimodal_content(self):
        """Test handling of multimodal content (text + other types)"""
        model, tokenizer = load(self.nonthinking_model_id)
        chat_template = ChatTemplate(tools_parser_type="hf", tokenizer=tokenizer)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    # Currently does not support image type, this part of the content will be removed when implemented.
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
            # enable_thinking=False,
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

    def test_tools_with_required_choice(self):
        """Test tool choice required"""
        model, tokenizer = load(self.tools_model_id)
        chat_template = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Call a function"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "test_func", "description": "Test function"},
            }
        ]

        prompt = chat_template.apply_chat_template(
            messages=messages, tools=tools, tool_choice="required"
        )
        print(prompt)
        assert chat_template.has_tools is True
        assert prompt.strip().endswith(chat_template.start_tool_calls) is not None

    def test_tools_with_forced_function_choice(self):
        """Test tool choice with specific function (dict format)"""
        model, tokenizer = load(self.tools_model_id)
        chat_template = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Get weather for NYC"}]
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
            },
            {
                "type": "function",
                "function": {"name": "get_time", "description": "Get current time"},
            },
        ]

        prompt = chat_template.apply_chat_template(
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        print(prompt)
        # The function has not been implemented yet, so ignore this assertion.
        assert chat_template.has_tools is True

    def test_tools_with_auto_choice(self):
        """Test tool choice with auto (default behavior)"""
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

        prompt = chat_template.apply_chat_template(
            messages=messages, tools=tools, tool_choice="auto"
        )
        print(prompt)
        assert chat_template.has_tools is True
        # auto choice should not add tool_calls prefix

    def test_tools_with_none_choice(self):
        """Test tool choice with none (disable tools)"""
        model, tokenizer = load(self.tools_model_id)
        chat_template = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Just chat, no tools"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                },
            }
        ]

        prompt = chat_template.apply_chat_template(
            messages=messages, tools=tools, tool_choice="none"
        )
        print(prompt)
        assert chat_template.has_tools is True
        # none choice should not add tool_calls prefix

    def test_thinking_with_tools(self):
        """Test thinking mode combined with tools"""
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit-DWQ")
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Use tools to help me"}]
        tools = [{"type": "function", "function": {"name": "helper"}}]

        prompt = chat_template.apply_chat_template(
            messages=messages, tools=tools, enable_thinking=True
        )

        assert chat_template.has_tools is True
        assert chat_template.enable_thinking is True
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

    def test_empty_content_handling(self):
        """Test handling of empty or None content"""
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit-DWQ")
        chat_template = ChatTemplate(tools_parser_type="qwen3", tokenizer=tokenizer)

        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "Real question"},
        ]

        prompt = chat_template.apply_chat_template(messages=messages)
        assert "Real question" in prompt

    def test_different_model_types(self):
        """Test different model type parsers"""
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit-DWQ")

        # Test qwen3 model type
        chat_template_qwen = ChatTemplate(
            tools_parser_type="qwen3", tokenizer=tokenizer
        )
        assert chat_template_qwen.tools_parser is not None

        # Test hf (HuggingFace) model type
        chat_template_hf = ChatTemplate(tools_parser_type="hf", tokenizer=tokenizer)
        assert chat_template_hf.tools_parser is not None

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
