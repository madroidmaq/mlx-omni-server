import json
import logging

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


class TestStructuredOutput:

    def test_structured_output_with_json_schema(self, openai_client):
        """Test structured generation with a JSON schema."""
        prompt = "List three colors and their hex codes."
        model_name = "mlx-community/gemma-3-1b-it-4bit-DWQ"

        json_schema = {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "hex": {
                                    "type": "string",
                                    "pattern": "^#[0-9A-Fa-f]{6}$",
                                },
                            },
                            "required": ["name", "hex"],
                        },
                    }
                },
                "required": ["colors"],
            },
        }

        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": json_schema},
            )
            logger.info(f"Chat Completion Response:\n{response}\n")

            # Validate response
            assert response.choices[0].message is not None, "No message in response"

            # Get generated content
            generated_content = response.choices[0].message.content
            logger.info(f"Generated content: {generated_content}")

            # Validate JSON format
            try:
                generated_json = json.loads(generated_content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON content: {e}")
                raise

            # Validate JSON structure
            assert "colors" in generated_json, "Missing colors field in JSON"
            assert isinstance(
                generated_json["colors"], list
            ), "Colors field is not an array"
            logger.info("Test passed: Generated JSON matches expected format")

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise

    def test_structured_output_with_beta(self, openai_client):
        from pydantic import BaseModel

        class Color(BaseModel):
            name: str
            hex: str

        class ColorsModel(BaseModel):
            colors: list[Color]

        prompt = "List three colors and their hex codes."
        model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"

        try:
            response = openai_client.beta.chat.completions.parse(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=ColorsModel,
            )

            logger.info(f"Chat Completion Response:\n{response}\n")

            # Validate response
            assert response.choices[0].message is not None, "No message in response"

            # Get generated content
            colors = response.choices[0].message.parsed
            logger.info(f"Generated content: {colors}")

            # Validate JSON structure
            assert colors is not None, "Colors is None"
            assert colors.colors is not None, "Colors list is None"
            assert colors.colors[0].name is not None, "Color name is None"
            assert colors.colors[0].hex is not None, "Color hex is None"

        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            raise
