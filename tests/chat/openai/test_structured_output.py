import json
import logging
from textwrap import dedent

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import BaseModel

from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from pydantic import BaseModel


@pytest.fixture
def client():
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=TestClient(app),
    )


class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


class TestStructuredOutput:

    def test_structured_output_with_json_schema(self, client):
        """
        Test structured generation with a JSON schema.
        https://cookbook.openai.com/examples/structured_outputs_intro
        """
        math_tutor_prompt = """
            You are a helpful math tutor. You will be provided with a math problem,
            and your goal will be to output a step by step solution, along with a final answer.
            For each step, just provide the output as an equation use the explanation field to detail the reasoning.
        """
        model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"

        # Use a simpler JSON schema approach
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "math_reasoning",
                "schema": {
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
                },
                "strict": True,
            },
        }

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": dedent(math_tutor_prompt)},
                {"role": "user", "content": "how can I solve 8x + 7 = -23"},
            ],
            response_format=response_format,
        )
        print(response)

        message = response.choices[0].message
        assert message.content is not None

        data = json.loads(message.content)
        assert data["steps"] is not None
        assert data["final_answer"] is not None

    def test_structured_output_with_pydantic(self, client):
        """
        Test structured generation with a JSON schema.
        https://cookbook.openai.com/examples/structured_outputs_intro
        """
        math_tutor_prompt = """
            You are a helpful math tutor. You will be provided with a math problem,
            and your goal will be to output a step by step solution, along with a final answer.
            For each step, just provide the output as an equation use the explanation field to detail the reasoning.
        """
        model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"

        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": dedent(math_tutor_prompt)},
                {"role": "user", "content": "how can I solve 8x + 7 = -23"},
            ],
            response_format=MathReasoning,
        )
        print(response)

        data = response.choices[0].message.parsed
        print(f"data: {data}")

        assert data.steps is not None
        assert data.final_answer is not None
