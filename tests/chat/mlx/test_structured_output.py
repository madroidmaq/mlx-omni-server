import json
import logging

from pydantic import BaseModel

from mlx_omni_server.chat.mlx.mlx_generate_wrapper import MLXGenerateWrapper
from mlx_omni_server.chat.mlx.model_types import MlxModelCache, ModelId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for testing
class Color(BaseModel):
    name: str
    hex: str


class ColorsResponse(BaseModel):
    colors: list[Color]


class UserProfile(BaseModel):
    name: str
    age: int
    email: str


class TestStructuredOutput:

    def test_structured_output_with_json_schema(self):
        """Test structured generation with a JSON schema."""
        # Try to load reasoning model
        model = MlxModelCache(model_id=ModelId(name="mlx-community/Qwen3-0.6B-4bit"))
        wrapper = MLXGenerateWrapper(model)

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

        result = wrapper.generate(
            messages=messages, template_kwargs=template_kwargs, json_schema=json_schema
        )
        print(result)

        assert result.text is not None, ""
        colors_data = json.loads(result.text)
        assert colors_data["colors"] is not None

    def test_structured_output_with_pydantic_model(self):
        """Test structured generation with a Pydantic BaseModel class."""
        # Try to load reasoning model
        model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
        model = MlxModelCache(model_id=ModelId(name=model_name))
        wrapper = MLXGenerateWrapper(model)

        messages = [
            {
                "role": "user",
                "content": "List three colors and their hex codes. Return only valid JSON.",
            }
        ]
        template_kwargs = {"enable_thinking": True}

        # Use Pydantic model as schema
        result = wrapper.generate(
            messages=messages,
            template_kwargs=template_kwargs,
            json_schema=ColorsResponse,
        )
        assert result.text is not None, ""
        colors_data = json.loads(result.text)
        assert colors_data["colors"] is not None

        # Validate that the response matches the Pydantic model structure
        colors_response = ColorsResponse.model_validate(colors_data)
        assert len(colors_response.colors) > 0
        assert all(
            isinstance(color.name, str) and isinstance(color.hex, str)
            for color in colors_response.colors
        )

    def test_structured_output_with_json_string(self):
        """Test structured generation with a JSON schema string."""
        # Try to load reasoning model
        model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
        model = MlxModelCache(model_id=ModelId(name=model_name))
        wrapper = MLXGenerateWrapper(model)

        messages = [
            {
                "role": "user",
                "content": "Create a user profile with name, age, and email. Return only valid JSON.",
            }
        ]
        template_kwargs = {"enable_thinking": True}

        # Use JSON schema string
        json_schema_str = """{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string"}
            },
            "required": ["name", "age", "email"]
        }"""

        result = wrapper.generate(
            messages=messages,
            template_kwargs=template_kwargs,
            json_schema=json_schema_str,
        )
        assert result.text is not None, ""
        user_data = json.loads(result.text)
        assert user_data["name"] is not None
        assert user_data["age"] is not None
        assert user_data["email"] is not None

        # Validate against our Pydantic model
        user_profile = UserProfile.model_validate(user_data)
        assert isinstance(user_profile.name, str)
        assert isinstance(user_profile.age, int)
        assert isinstance(user_profile.email, str)
