import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


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
