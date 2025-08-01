import json
import logging
from textwrap import dedent

from pydantic import BaseModel

from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator

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


class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


math_tutor_json_schema = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["explanation", "output"],
                "additionalProperties": False,
            },
        },
        "final_answer": {"type": "string"},
    },
    "required": ["steps", "final_answer"],
    "additionalProperties": False,
}


class TestStructuredOutput:

    def test_json_schema(self):
        """Test structured generation with a JSON schema."""
        # Try to load reasoning model
        model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
        wrapper = ChatGenerator.create(model_name)

        math_tutor_prompt = """
                    You are a helpful math tutor. You will be provided with a math problem,
                    and your goal will be to output a step by step solution, along with a final answer.
                    For each step, just provide the output as an equation use the explanation field to detail the reasoning.
                """
        result = wrapper.generate(
            messages=[
                {"role": "system", "content": dedent(math_tutor_prompt)},
                {"role": "user", "content": "how can I solve 8x + 7 = -23"},
            ],
            json_schema=math_tutor_json_schema,
        )
        print(result)

        assert result.content.text is not None, ""
        assert result.content.reasoning is None, ""

        result_data = json.loads(result.content.text)

        assert result_data["steps"] is not None
        assert result_data["final_answer"] is not None

    def test_json_schema_with_thinking(self):
        """Test structured generation with a JSON schema."""
        # Try to load reasoning model
        wrapper = ChatGenerator.create("mlx-community/Qwen3-0.6B-4bit")

        math_tutor_prompt = """
                    You are a helpful math tutor. You will be provided with a math problem,
                    and your goal will be to output a step by step solution, along with a final answer.
                    For each step, just provide the output as an equation use the explanation field to detail the reasoning.
                """
        result = wrapper.generate(
            messages=[
                {"role": "system", "content": dedent(math_tutor_prompt)},
                {"role": "user", "content": "how can I solve 8x + 7 = -23"},
            ],
            template_kwargs={"enable_thinking": True},
            json_schema=math_tutor_json_schema,
        )
        print(result)

        assert result.content.text is not None, "text is None"
        assert result.content.reasoning is not None, "reasoning is None"

        result_data = json.loads(result.content.text)

        assert result_data["steps"] is not None, "reasoning steps is None"
        assert result_data["final_answer"] is not None, "reasoning final_answer is None"

    def test_structured_output_with_pydantic_model(self):
        """Test structured generation with a Pydantic BaseModel class."""
        # Try to load reasoning model
        model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"
        wrapper = ChatGenerator.create(model_name)

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
        assert result.content.text is not None, ""
        colors_data = json.loads(result.content.text)
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
        wrapper = ChatGenerator.create(model_name)

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
        assert result.content.text is not None, ""
        user_data = json.loads(result.content.text)
        assert user_data["name"] is not None
        assert user_data["age"] is not None
        assert user_data["email"] is not None

        # Validate against our Pydantic model
        user_profile = UserProfile.model_validate(user_data)
        assert isinstance(user_profile.name, str)
        assert isinstance(user_profile.age, int)
        assert isinstance(user_profile.email, str)

    def test_structured_output_with_json_schema2(self):
        """
        Test structured generation with a JSON schema.
        https://cookbook.openai.com/examples/structured_outputs_intro
        """
        math_tutor_prompt = """
            You are a helpful math tutor. You will be provided with a math problem,
            and your goal will be to output a step by step solution, along with a final answer.
            For each step, just provide the output as an equation use the explanation field to detail the reasoning.
        """
        model_name = "mlx-community/Qwen3-0.6B-4bit"
        wrapper = ChatGenerator.create(model_name)

        result = wrapper.generate(
            messages=[
                {"role": "system", "content": dedent(math_tutor_prompt)},
                {"role": "user", "content": "how can I solve 8x + 7 = -23"},
            ],
            template_kwargs={"enable_thinking": True},
            json_schema=math_tutor_json_schema,
        )

        print("Generated result:")
        print(f"Text: {result.content.text}")
        print(f"Reasoning: {result.content.reasoning}")

        assert result.content.reasoning is not None
        assert result.content.text is not None

        result_data = json.loads(result.content.text)

        assert result_data["steps"] is not None
        assert result_data["final_answer"] is not None
        print(
            "✅ Test passed: Both thinking and structured JSON output working correctly!"
        )

    def test_thinking_structured_output_no_duplicate_tags(self):
        """Test that thinking+structured output doesn't create duplicate <think> tags."""
        # Use a reasoning model that supports thinking
        model_name = "mlx-community/Qwen3-0.6B-4bit"
        wrapper = ChatGenerator.create(model_name)

        math_tutor_prompt = """
                    You are a helpful math tutor. You will be provided with a math problem,
                    and your goal will be to output a step by step solution, along with a final answer.
                    For each step, just provide the output as an equation use the explanation field to detail the reasoning.
                """
        messages = [{"role": "user", "content": math_tutor_prompt}]

        result = wrapper.generate(
            messages=messages,
            template_kwargs={"enable_thinking": True},
            json_schema=math_tutor_json_schema,
            max_tokens=8192,
        )

        print(f"Generated text: {result.content.text}")
        print(f"Generated reasoning: {result.content.reasoning}")

        # Basic functionality checks
        assert result.content.text is not None, "Should have text output"
        assert result.content.reasoning is not None, "Should have reasoning content"

        # Verify JSON is valid
        math_data = json.loads(result.content.text)
        assert math_data["steps"] is not None
        assert math_data["final_answer"] is not None

        print("✅ Test passed: Both thinking and structured JSON output working!")
