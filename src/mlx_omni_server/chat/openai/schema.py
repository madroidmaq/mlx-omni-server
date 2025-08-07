import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator


class ToolType(str, Enum):
    FUNCTION = "function"


class FunctionParameters(BaseModel):
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None


class Function(BaseModel):
    name: str = Field(..., max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = None
    parameters: Optional[FunctionParameters] = None


class Tool(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: Function


class ToolChoice(str, Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class SpecificToolChoice(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: Dict[str, str]


class FunctionCall(BaseModel):
    """Function call details within a tool call."""

    name: str
    arguments: str  # JSON string of arguments


class ToolCall(BaseModel):
    """Tool call from model output."""

    id: str
    type: ToolType = ToolType.FUNCTION
    function: FunctionCall

    @classmethod
    def from_llama_output(
        cls, name: str, parameters: Dict[str, Any], call_id: str
    ) -> "ToolCall":
        """Create a ToolCall instance from Llama model output format."""
        return cls(
            id=call_id,
            type=ToolType.FUNCTION,
            function=FunctionCall(
                name=name,
                arguments=json.dumps(parameters),  # Convert parameters to JSON string
            ),
        )


ToolChoiceType = Union[ToolChoice, SpecificToolChoice]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: Role
    content: Optional[Union[str, List[Dict[str, str]]]] = None
    reasoning: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    class Config:
        json_encoders = {bytes: lambda v: v.decode()}


class ChatCompletionUsageDetails(BaseModel):
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class PromptTokensDetails(BaseModel):
    """包含提示令牌的详细信息"""

    cached_tokens: int = 0


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[PromptTokensDetails] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[ChatCompletionUsage] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class JsonSchemaFormat(BaseModel):
    description: Optional[str] = Field(
        None, description="A description of what the response format is for"
    )
    name: str = Field(
        ...,
        description="The name of the response format",
        pattern="^[a-zA-Z0-9_-]{1,64}$",
    )
    schema_def: Optional[Dict[str, Any]] = Field(
        None, description="The schema for the response format", alias="schema"
    )
    strict: Optional[bool] = Field(
        False, description="Whether to enable strict schema adherence"
    )

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        if len(v) > 64:
            raise ValueError("Name must not exceed 64 characters")
        if not re.match("^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Name must only contain a-z, A-Z, 0-9, underscores and dashes"
            )
        return v


class ResponseFormat(BaseModel):
    type: str = Field(..., description="The type of response format")
    json_schema: Optional[JsonSchemaFormat] = Field(
        None, description="The JSON schema configuration when type is 'json_schema'"
    )

    @field_validator("type")
    def validate_type(cls, v):
        if v not in ["text", "json_object", "json_schema"]:
            raise ValueError(
                "Type must be one of: 'text', 'json_object', or 'json_schema'"
            )
        return v

    @field_validator("json_schema")
    def validate_json_schema(cls, v, values):
        type_val = values.data.get("type")
        if type_val == "json_schema" and v is None:
            raise ValueError("json_schema is required when type is 'json_schema'")
        if type_val != "json_schema" and v is not None:
            raise ValueError(
                "json_schema should only be provided when type is 'json_schema'"
            )
        return v


class ChatCompletionRequest(BaseModel):
    # Standard OpenAI API fields
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(1.0, ge=0, le=2)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = Field(
        None,
        ge=0,
        le=20,
    )
    n: Optional[int] = Field(1, ge=1, le=10)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoiceType] = None
    response_format: Optional[ResponseFormat] = None

    # Allow any additional fields
    class Config:
        extra = "allow"  # This allows additional fields not defined in the model

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    def validate_top_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields: Set[str] = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream_options",
            "response_format",
            "user",
            "metadata",
            "modalities",
            "store",
            "draft-model",
        }
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}
