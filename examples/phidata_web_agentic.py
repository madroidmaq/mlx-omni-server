from openai import OpenAI
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

# Use mlx-omni-server to provide local OpenAI service
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="mlx-omni-server",  # not-needed
)

web_agent = Agent(
    model=OpenAIChat(
        client=client,
        id="mlx-community/Qwen2.5-3B-Instruct-4bit",
    ),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)
web_agent.print_response("Tell me about Apple MLX?", stream=False)
