import unittest

from mlx_omni_server.chat.mlx.models import load_model
from mlx_omni_server.chat.schema import (
    ChatCompletionRequest,
    ChatMessage,
    ResponseFormat,
    Role,
)


class TestStructuredOutput(unittest.TestCase):

    def test_structured_output_with_json_schema(self):
        """Test structured generation with a JSON schema."""
        prompt = "List three colors and their hex codes."
        model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"

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

        request = ChatCompletionRequest(
            model=model_name,
            messages=[ChatMessage(role=Role.USER, content=prompt)],
            response_format=ResponseFormat(type="json_schema", json_schema=json_schema),
        )

        model = load_model(request.model)
        response = model.generate(request)

        print(f"Chat Completion Response:\n{response}\n")
        print(f"Chat Completion Response Message:\n{response.choices[0].message}\n")

        # Basic validation that the output looks like JSON
        self.assertTrue(response.choices[0].message.content.strip().startswith("{"))
        self.assertTrue(response.choices[0].message.content.strip().endswith("}"))
        self.assertIn("colors", response.choices[0].message.content)
        self.assertIn("name", response.choices[0].message.content)
        self.assertIn("hex", response.choices[0].message.content)
