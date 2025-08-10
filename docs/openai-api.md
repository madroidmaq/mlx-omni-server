<div align="center">

# OpenAI API Documentation

*Complete OpenAI API compatibility for local MLX inference*

[Installation](#-installation--setup) • [Quick Start](#-basic-usage) • [Endpoints](#-supported-endpoints) • [Advanced Usage](#-advanced-usage)

</div>

---

MLX Omni Server provides full OpenAI API compatibility, enabling seamless integration with existing OpenAI SDK clients while leveraging local MLX inference on Apple Silicon.

## 🚀 Installation & Setup

```bash
pip install mlx-omni-server
mlx-omni-server  # Start the server
```

## ⚡ Basic Usage

```python
from openai import OpenAI

# Connect to local server
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
)

# Simple chat completion
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## 📋 Supported Endpoints

| Endpoint | Feature | Status |
|----------|---------|--------|
| `/v1/chat/completions` | Chat with tools, streaming, structured output | ✅ |
| `/v1/audio/speech` | Text-to-Speech generation | ✅ |
| `/v1/audio/transcriptions` | Speech-to-Text transcription | ✅ |
| `/v1/images/generations` | Image generation from text prompts | ✅ |
| `/v1/embeddings` | Text embedding generation | ✅ |
| `/v1/models` | Model listing and management | ✅ |

### Chat Completions

#### Basic Chat Completion

```python
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500
)
```

#### Streaming Chat Completion

```python
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)
```

#### Structured Output

```python
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: float

response = client.chat.completions.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Get weather for New York"}],
    response_format={"type": "json_object", "schema": WeatherResponse.model_json_schema()}
)
```

### Audio Processing

#### Text-to-Speech (`/v1/audio/speech`)

```python
speech_file_path = "output.wav"
response = client.audio.speech.create(
    model="lucasnewman/f5-tts-mlx",
    voice="alloy",
    input="Hello from MLX Omni Server!"
)
response.stream_to_file(speech_file_path)
```

#### Speech-to-Text (`/v1/audio/transcriptions`)

```python
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="mlx-community/whisper-large-v3-turbo",
        file=audio_file
    )
    print(transcript.text)
```

### Image Generation (`/v1/images/generations`)

```python
response = client.images.generate(
    model="argmaxinc/mlx-FLUX.1-schnell",
    prompt="A serene landscape with mountains and a lake at sunset",
    n=1,
    size="1024x1024"
)

# Save the generated image
image_url = response.data[0].url
print(f"Generated image: {image_url}")
```

### Embeddings (`/v1/embeddings`)

```python
# Single text embedding
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="MLX Omni Server provides local AI inference"
)
print(f"Embedding dimension: {len(response.data[0].embedding)}")

# Multiple text embeddings
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=["Hello world", "Machine learning is fascinating"]
)
```

### Models (`/v1/models`)

```python
# List all available models
models = client.models.list()
for model in models.data:
    print(f"{model.id} - Created: {model.created}")

# Get specific model info
model = client.models.retrieve("mlx-community/gemma-3-1b-it-4bit-DWQ")
print(f"Model details: {model}")
```

## 🔧 Advanced Usage

### Model Management

#### Using Local Models

```python
# Use a model from your local filesystem
response = client.chat.completions.create(
    model="/path/to/your/local/model",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Model Caching

The server automatically caches models to improve performance:

```python
# First request (slower - model loading/downloading)
response1 = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "First request"}]
)

# Subsequent requests (faster - using cached model)
response2 = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Second request"}]
)
```

### Advanced Configuration

#### Custom Parameters

```python
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Generate creative content"}],
    temperature=0.8,        # Higher temperature for more creativity
    top_p=0.9,             # Nucleus sampling
    max_tokens=1000,       # Maximum response length
    presence_penalty=0.1,   # Encourage new topics
    frequency_penalty=0.1  # Discourage repetition
)
```

#### Batch Processing

```python
# Process multiple requests efficiently
requests = [
    {"role": "user", "content": f"Explain {topic}"}
    for topic in ["AI", "ML", "Deep Learning"]
]

for request in requests:
    response = client.chat.completions.create(
        model="mlx-community/gemma-3-1b-it-4bit-DWQ",
        messages=[request]
    )
    print(f"Response: {response.choices[0].message.content}")
```

## 📡 REST API Examples

### Chat Completions

```bash
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Streaming Chat

```bash
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [
      {"role": "user", "content": "Tell me a joke"}
    ],
    "stream": true
  }'
```

### Text-to-Speech

```bash
curl -X POST "http://localhost:10240/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lucasnewman/f5-tts-mlx",
    "input": "Hello from MLX!",
    "voice": "alloy"
  }' \
  --output speech.wav
```

### Image Generation

```bash
curl http://localhost:10240/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "argmaxinc/mlx-FLUX.1-schnell",
    "prompt": "A beautiful sunset over mountains",
    "n": 1,
    "size": "1024x1024"
  }'
```

## 🧪 Development & Testing

### Using TestClient

For development without running a server:

```python
from openai import OpenAI
from fastapi.testclient import TestClient
from mlx_omni_server.main import app

# Use TestClient directly
client = OpenAI(http_client=TestClient(app))

response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Error Handling

```python
try:
    response = client.chat.completions.create(
        model="non-existent-model",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    print(f"Error: {e}")
```

## 📊 Performance Tips

1. **Model Selection**: Use smaller models for faster inference
2. **Caching**: Reuse models across requests for better performance
3. **Streaming**: Use streaming for long responses to improve perceived performance
4. **Batch Processing**: Process multiple requests together when possible
5. **Temperature**: Lower temperature (0.1-0.3) for faster, more focused responses

## 🔍 Troubleshooting

### Common Issues

**Model Download Takes Too Long**
```bash
# Pre-download models using HuggingFace CLI
huggingface-cli download mlx-community/gemma-3-1b-it-4bit-DWQ
```

**Server Won't Start**
```bash
# Check Python version (requires 3.9+)
python --version

# Check MLX installation
python -c "import mlx; print(mlx.__version__)"
```

**Memory Issues**
```bash
# Use smaller models for devices with limited memory
mlx-community/gemma-2b-it-4bit-DWQ
```

### Debug Mode

```bash
# Start server with debug logging
MLX_OMNI_LOG_LEVEL=debug mlx-omni-server
```

## 📚 API Reference

For complete API specifications, see:
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [MLX Omni Server Source Code](https://github.com/madroidmaq/mlx-omni-server)

## 🤝 Contributing

Contributions are welcome! Please see the main repository for guidelines on:
- Setting up development environment
- Running tests
- Submitting pull requests

---

**Note**: This documentation covers OpenAI API compatibility. For Anthropic API documentation, see [docs/anthropic-api.md](../anthropic-api.md).
