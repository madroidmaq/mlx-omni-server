import json
import logging

from mlx_omni_server.chat.mlx.mlx_generate_wrapper import MLXGenerateWrapper
from mlx_omni_server.chat.mlx.model_types import MlxModelCache, ModelId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStructuredOutput:

    def test_structured_output_with_json_schema(self):
        """Test structured generation with a JSON schema."""
        # Try to load reasoning model
        reasoning_model = MlxModelCache(
            model_id=ModelId(name="mlx-community/Qwen3-0.6B-4bit")
        )
        reasoning_wrapper = MLXGenerateWrapper(reasoning_model)

        messages = [
            {
                "role": "user",
                "content": "List three colors and their hex codes. Return only valid JSON.",
            }
        ]
        template_kwargs = {"enable_thinking": True}

        # Use a simpler JSON schema approach
        json_schema = {
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "hex": {"type": "string"},
                        },
                        "required": ["name", "hex"],
                    },
                }
            },
            "required": ["colors"],
        }

        result = reasoning_wrapper.generate(
            messages=messages, template_kwargs=template_kwargs, json_schema=json_schema
        )
        assert result.text is not None, ""
        colors_data = json.loads(result.text)
        assert colors_data["colors"] is not None
