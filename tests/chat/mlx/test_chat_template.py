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
        # auto choice should not add tool_calls prefix

        # Test none choice (should not add prefix)
        chat_template3 = ChatTemplate(tools_parser_type="llama", tokenizer=tokenizer)
        prompt_none = chat_template3.apply_chat_template(
            messages=messages, tools=tools, tool_choice="none"
        )
        assert chat_template3.has_tools is True
        # none choice should not add tool_calls prefix

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
