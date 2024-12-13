import asyncio
import json

import weave
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

weave.init("mlx-omni-function-calling-benchmark")

client = OpenAI(
    base_url="http://localhost:10240/v1",
    # base_url="http://localhost:11434/v1",
    api_key="mlx-omni-server",  # not-needed
)


class FunctionCallingModel(weave.Model):
    model_name: str

    @weave.op()
    def predict(self, messages, tools, tool_calls, target_name) -> dict:
        response = client.chat.completions.create(
            model=self.model_name, messages=messages, tools=tools, tool_choice="auto"
        )
        message = response.choices[0].message
        if not tool_calls:
            return {
                "expected": [],
                "function_name": None,
                "successful": False,
                "message": "no expected tool calls",
            }

        # function_name =target_name
        if not message.tool_calls:
            return {
                "expected": tool_calls,
                "function_name": target_name,
                "successful": False,
                "is_tool_call": False,
                "content": message.content,
                "message": f"not a tool call, message content: {message.content}",
            }

        actual_calls = message.tool_calls[0]
        return {
            "expected": tool_calls,
            "actual_calls": actual_calls,
            "is_tool_call": True,
            "function_name": target_name,
            "successful": actual_calls.function.name == target_name,
            "message": f"gen tool calls, expected {target_name} but actual calls is {actual_calls.function.name}",
        }


@weave.op()
def tool_call_score(output: dict) -> dict:
    correct = "successful" in output and output["successful"]
    is_tool_call = output["is_tool_call"]

    return {"is_matched": correct, "is_tool_call": is_tool_call}


def run_eval(model_name: str):
    dataset = load_dataset("madroid/glaive-function-calling-openai", split="test")

    examples = []
    for i, example in enumerate(
        tqdm(dataset, desc="Processing examples", unit="example")
    ):
        data = json.loads(example["json"])
        if "tool_calls" in data:
            expected_calls = data.get("tool_calls", [])
            target_name = expected_calls[0]["function"]["name"]
            examples.append(
                {
                    "id": str(i),
                    "messages": data["messages"],
                    "tools": data["tools"],
                    "tool_calls": data["tool_calls"],
                    "target_name": target_name,
                }
            )

    evaluation = weave.Evaluation(
        name="function_call_eval",
        dataset=examples,
        scorers=[tool_call_score],
    )

    model = FunctionCallingModel(name="my_func_call_model", model_name=model_name)

    results = asyncio.run(evaluation.evaluate(model))
    print(results)


if __name__ == "__main__":
    model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # model_name = "llama3.2:3b"
    run_eval(model_name)
