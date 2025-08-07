import logging

import anthropic
import pytest
from fastapi.testclient import TestClient

from mlx_omni_server.chat.anthropic.anthropic_schema import MessagesRequest
from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def anthropic_client(client):
    """Create Anthropic client configured with test server and handle cache cleanup."""
    # The test will use this client instance
    yield anthropic.Anthropic(
        base_url="http://test/anthropic",
        api_key="not-needed",
        http_client=client,
    )

    # Teardown logic: runs after the test is finished
    # This clears the global model cache to prevent state pollution between tests
    import mlx_omni_server.chat.anthropic.router as anthropic_router

    anthropic_router._cached_model = None
    anthropic_router._cached_anthropic_adapter = None


@pytest.fixture
def direct_client(client):
    """Direct HTTP client for testing raw API responses"""
    return client


class TestAnthropicMessages:
    """Test suite for Anthropic Messages API"""

    thinking_model = "Qwen/Qwen3-0.6B-MLX-4bit"
    model_id = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    max_tokens = 4096

    def test_messages_basic(self, anthropic_client):
        """Test basic message completion"""
        try:
            response = anthropic_client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Hello!"}],
            )
            logger.info(f"Anthropic Messages Response:\\n{response}\\n")

            # Validate response structure
            assert response.model == self.model_id, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.type == "message", "Response type is not 'message'"
            assert response.role == "assistant", "Response role is not 'assistant'"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"
            assert response.stop_reason is not None, "No stop reason in response"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_basic_text_block(self, anthropic_client):
        """Test basic message completion"""
        try:
            response = anthropic_client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                # messages=[{"role": "user", "content": "Hello!"}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Why is the ocean salty?"}
                        ],
                    }
                ],
            )
            logger.info(f"Anthropic Messages Response:\\n{response}\\n")

            # Validate response structure
            assert response.model == self.model_id, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.type == "message", "Response type is not 'message'"
            assert response.role == "assistant", "Response role is not 'assistant'"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"
            assert response.stop_reason is not None, "No stop reason in response"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_conversation(self, anthropic_client):
        """Test multi-turn conversation"""
        try:
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "Hi there!"},
                    {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    {"role": "user", "content": "Can you explain what AI is?"},
                ],
            )
            logger.info(f"Conversation Response:\\n{response}\\n")
            logger.info(f"Conversation Usage:\\n{response.usage}\\n")

            # Validate response
            assert response.model == model, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_with_system_prompt(self, anthropic_client):
        """Test message completion with system prompt"""
        try:
            response = anthropic_client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system="You are a helpful assistant that responds in a friendly manner.",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )
            logger.info(f"System Prompt Response:\\n{response}\\n")

            # Validate response
            assert response.model == self.model_id, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_stream(self, anthropic_client):
        """Test streaming message completion using anthropic_client"""
        try:
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"

            # Validate streaming response
            event_count = 0
            content_text = ""
            message_start_received = False
            message_delta_received = False
            message_stop_received = False
            content_block_start_received = False
            content_block_stop_received = False
            text_deltas_received = 0

            with anthropic_client.messages.stream(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Count from 1 to 5."}],
            ) as stream:
                for event in stream:
                    event_count += 1

                    if event.type == "message_start":
                        message_start_received = True
                        assert (
                            event.message.model == model
                        ), "Incorrect model name in stream"

                    elif event.type == "content_block_start":
                        content_block_start_received = True
                        assert (
                            event.content_block is not None
                        ), "Missing content_block in content_block_start"
                        assert (
                            event.index is not None
                        ), "Missing index in content_block_start"

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # Log all delta details for debugging
                        logger.info(f"Delta details: type={delta.type}, delta={delta}")
                        if delta.type == "text_delta" and delta.text:
                            content_text += delta.text
                            text_deltas_received += 1
                        elif delta.type == "thinking_delta" and hasattr(
                            delta, "thinking"
                        ):
                            # This might happen if model unexpectedly switches to thinking mode
                            logger.warning(
                                f"Received unexpected thinking_delta: {delta.thinking[:50]}..."
                            )
                        elif delta.type:
                            logger.warning(
                                f"Received unexpected delta type: {delta.type}"
                            )
                        else:
                            logger.error(f"Delta missing type field: {delta}")

                    elif event.type == "content_block_stop":
                        content_block_stop_received = True
                        assert (
                            event.index is not None
                        ), "Missing index in content_block_stop"

                    elif event.type == "message_delta":
                        message_delta_received = True
                        delta = event.delta
                        assert (
                            delta.stop_reason is not None
                        ), "No stop reason in message_delta"
                        # Usage should be available in the event
                        assert event.usage is not None, "No usage in message_delta"

                    elif event.type == "message_stop":
                        message_stop_received = True

            # Validate overall streaming response
            assert event_count > 0, "No stream events received"
            assert message_start_received, "No message_start event received"
            assert content_block_start_received, "No content_block_start event received"
            assert content_block_stop_received, "No content_block_stop event received"
            assert message_delta_received, "No message_delta event received"
            assert message_stop_received, "No message_stop event received"
            assert (
                text_deltas_received > 0
            ), f"No text deltas received (got {event_count} total events)"
            assert content_text.strip(), "Generated content is empty"
            logger.info(f"Complete generated content: {content_text}")
            logger.info(f"Received {text_deltas_received} text delta events")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_thinking_stream(self, anthropic_client):
        """Test streaming message completion with thinking mode enabled"""
        try:

            # Validate streaming response with thinking
            event_count = 0
            thinking_content = ""
            text_content = ""
            message_start_received = False
            message_delta_received = False
            message_stop_received = False
            thinking_deltas_received = 0
            text_deltas_received = 0
            signature_delta_received = False

            with anthropic_client.messages.stream(
                model=self.thinking_model,
                max_tokens=self.max_tokens,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[
                    {
                        "role": "user",
                        "content": "Solve this step by step: What is 15 + 27?",
                    }
                ],
            ) as stream:
                for event in stream:
                    event_count += 1

                    if event.type == "message_start":
                        message_start_received = True

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "thinking_delta" and hasattr(
                            delta, "thinking"
                        ):
                            thinking_content += delta.thinking
                            thinking_deltas_received += 1
                        elif delta.type == "text_delta" and delta.text:
                            text_content += delta.text
                            text_deltas_received += 1
                        elif delta.type == "signature_delta":
                            signature_delta_received = True

                    elif event.type == "message_delta":
                        message_delta_received = True

                    elif event.type == "message_stop":
                        message_stop_received = True

            # Validate overall streaming response
            assert event_count > 0, "No stream events received"
            assert message_start_received, "No message_start event received"
            assert message_delta_received, "No message_delta event received"
            assert message_stop_received, "No message_stop event received"

            # Check that we received either thinking or text content (or both)
            assert (
                thinking_deltas_received > 0 or text_deltas_received > 0
            ), "No content deltas received"

            if thinking_deltas_received > 0:
                assert thinking_content.strip(), "Thinking content is empty"
                logger.info(f"Thinking content: {thinking_content}")
                logger.info(
                    f"Received {thinking_deltas_received} thinking delta events"
                )
                # If we had thinking content, we should have received signature delta
                # Note: signature_delta is only sent if there was thinking content

            if text_deltas_received > 0:
                assert text_content.strip(), "Text content is empty"
                logger.info(f"Text content: {text_content}")
                logger.info(f"Received {text_deltas_received} text delta events")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_stream_event_order(self, anthropic_client):
        """验证流式事件的严格顺序：message_start → content_block_start → deltas → content_block_stop → message_delta → message_stop"""
        try:
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"

            # 严格验证事件顺序
            events_order = []

            with anthropic_client.messages.stream(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Say hello"}],
            ) as stream:
                for event in stream:
                    event_type = event.type
                    if event_type:
                        events_order.append(event_type)

            # 验证事件顺序
            logger.info(f"Received events in order: {events_order}")

            # 检查必需的事件都存在
            assert len(events_order) > 0, "No events received at all"
            assert (
                "message_start" in events_order
            ), f"Missing message_start event in {events_order}"
            assert (
                "content_block_start" in events_order
            ), f"Missing content_block_start event in {events_order}"
            assert (
                "content_block_stop" in events_order
            ), f"Missing content_block_stop event in {events_order}"
            assert (
                "message_delta" in events_order
            ), f"Missing message_delta event in {events_order}"
            assert (
                "message_stop" in events_order
            ), f"Missing message_stop event in {events_order}"
            assert (
                events_order.count("content_block_delta") > 0
            ), f"Missing content_block_delta events in {events_order}"

            # 验证严格顺序
            message_start_idx = events_order.index("message_start")
            content_block_start_idx = events_order.index("content_block_start")
            first_delta_idx = events_order.index("content_block_delta")
            content_block_stop_idx = events_order.index("content_block_stop")
            message_delta_idx = events_order.index("message_delta")
            message_stop_idx = events_order.index("message_stop")

            assert (
                message_start_idx < content_block_start_idx
            ), "message_start should come before content_block_start"
            assert (
                content_block_start_idx < first_delta_idx
            ), "content_block_start should come before first content_block_delta"
            assert (
                first_delta_idx < content_block_stop_idx
            ), "content_block_delta should come before content_block_stop"
            assert (
                content_block_stop_idx < message_delta_idx
            ), "content_block_stop should come before message_delta"
            assert (
                message_delta_idx < message_stop_idx
            ), "message_delta should come before message_stop"

            # 验证所有content_block_delta都在start和stop之间
            for i, event in enumerate(events_order):
                if event == "content_block_delta":
                    assert (
                        i > content_block_start_idx
                    ), f"content_block_delta at index {i} should come after content_block_start"
                    assert (
                        i < content_block_stop_idx
                    ), f"content_block_delta at index {i} should come before content_block_stop"

            # 验证message_start是第一个事件，message_stop是最后一个事件
            assert (
                events_order[0] == "message_start"
            ), f"message_start should be the first event, got: {events_order[0]}"
            assert (
                events_order[-1] == "message_stop"
            ), f"message_stop should be the last event, got: {events_order[-1]}"

            logger.info("✅ Event order validation passed")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_stream_thinking_then_text(self, anthropic_client):
        """测试先有thinking再有text的流式事件序列"""
        try:
            # 跟踪事件和内容块
            events_order = []
            content_blocks = {}  # index -> block_info
            thinking_content = ""
            text_content = ""

            with anthropic_client.messages.stream(
                model=self.thinking_model,
                max_tokens=self.max_tokens,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[
                    {
                        "role": "user",
                        "content": "Think step by step and then answer: What is 2+3?",
                    }
                ],
            ) as stream:
                for event in stream:
                    event_type = event.type

                    if event_type:
                        events_order.append(event_type)

                        if event_type == "content_block_start":
                            current_block_index = event.index
                            content_block = event.content_block
                            block_type = content_block.type if content_block else None
                            content_blocks[current_block_index] = {
                                "type": block_type,
                                "deltas": [],
                            }
                            logger.info(
                                f"Started content block {current_block_index} of type: {block_type}"
                            )

                        elif event_type == "content_block_delta":
                            delta = event.delta
                            delta_type = delta.type if delta else None
                            index = event.index

                            if index is not None and index in content_blocks:
                                content_blocks[index]["deltas"].append(delta_type)

                            if delta_type == "thinking_delta" and hasattr(
                                delta, "thinking"
                            ):
                                thinking_content += delta.thinking
                            elif delta_type == "text_delta" and delta.text:
                                text_content += delta.text

                        elif event_type == "content_block_stop":
                            index = event.index
                            if index is not None and index in content_blocks:
                                logger.info(
                                    f"Stopped content block {index}, deltas: {content_blocks[index]['deltas']}"
                                )

            logger.info(f"Events order: {events_order}")
            logger.info(f"Content blocks: {content_blocks}")
            logger.info(f"Thinking content: '{thinking_content}'")
            logger.info(f"Text content: '{text_content}'")

            # 验证基本事件存在
            assert "message_start" in events_order, "Missing message_start event"
            assert "message_delta" in events_order, "Missing message_delta event"
            assert "message_stop" in events_order, "Missing message_stop event"

            # 验证内容块结构
            assert len(content_blocks) > 0, "No content blocks found"

            # 如果有thinking内容，验证thinking block的存在和顺序
            if thinking_content.strip():
                # 找到thinking block
                thinking_block_found = False
                text_block_found = False
                thinking_block_index = None
                text_block_index = None

                for index, block_info in content_blocks.items():
                    if block_info["type"] == "thinking":
                        thinking_block_found = True
                        thinking_block_index = index
                        assert (
                            "thinking_delta" in block_info["deltas"]
                        ), f"Thinking block {index} should have thinking_delta"

                    elif block_info["type"] == "text":
                        text_block_found = True
                        text_block_index = index
                        assert (
                            "text_delta" in block_info["deltas"]
                        ), f"Text block {index} should have text_delta"

                if thinking_block_found:
                    logger.info(f"Found thinking block at index {thinking_block_index}")

                    # 如果同时有thinking和text，验证thinking在text之前
                    if text_block_found:
                        assert (
                            thinking_block_index < text_block_index
                        ), "Thinking block should come before text block"
                        logger.info(
                            f"Verified thinking block {thinking_block_index} comes before text block {text_block_index}"
                        )

                assert thinking_content.strip(), "Thinking content should not be empty"
                logger.info("✅ Thinking content validation passed")

            # 验证至少有一些内容（thinking或text）
            assert (
                thinking_content.strip() or text_content.strip()
            ), "Should have either thinking or text content"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_thinking_mode(self, anthropic_client):
        """Test message completion with thinking mode enabled"""
        try:
            model = self.thinking_model

            response = anthropic_client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                thinking={"type": "enabled", "budget_tokens": 1024},  # Must be >= 1024
                messages=[
                    {
                        "role": "user",
                        "content": "Solve this step by step: What is 15 + 27?",
                    }
                ],
            )

            logger.info(f"Thinking Mode Response:\\n{response}\\n")

            # Validate response structure
            assert response.model == model, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert len(response.content) > 0, "No content blocks in response"

            # Check if thinking content is present (might be in separate block)
            has_thinking = any(block.type == "thinking" for block in response.content)
            has_text = any(block.type == "text" for block in response.content)

            assert has_thinking, "No thinking content block found"
            assert has_text, "No text content block found"
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_error_handling(self, anthropic_client):
        """Test error handling for invalid requests"""
        try:
            # Test missing required field (max_tokens)
            with pytest.raises(
                Exception
            ):  # anthropic client will raise TypeError for missing required params
                anthropic_client.messages.create(
                    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
                    messages=[{"role": "user", "content": "Hello!"}],
                    # Missing max_tokens
                )

            logger.info("Error handling validation passed")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_messages_schema_validation(self):
        """Test Pydantic schema validation"""
        try:
            # Test valid request
            valid_request = {
                "model": "test-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello!"}],
            }

            request = MessagesRequest(**valid_request)
            assert request.model == "test-model"
            assert request.max_tokens == 100
            assert len(request.messages) == 1

            # Test invalid temperature
            with pytest.raises(ValueError):
                MessagesRequest(
                    model="test-model",
                    max_tokens=self.max_tokens,
                    temperature=2.0,  # Invalid: > 1.0
                    messages=[{"role": "user", "content": "Hello!"}],
                )

            logger.info("Schema validation tests passed")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_usage_tracking(self, anthropic_client):
        """Test that usage statistics are properly tracked"""
        try:
            model = self.model_id
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Hi"}],
            )

            # Validate usage statistics
            assert response.usage is not None, "No usage statistics"
            assert response.usage.input_tokens > 0, "No input tokens counted"
            assert response.usage.output_tokens > 0, "No output tokens counted"

            logger.info(
                f"Usage: {response.usage.input_tokens} input, {response.usage.output_tokens} output tokens"
            )

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise
