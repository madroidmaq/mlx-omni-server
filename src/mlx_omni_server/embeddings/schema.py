from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    BERT = "bert"


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    input: Union[str, List[str]] = Field(
        ..., description="Input text to get embeddings for"
    )
    encoding_format: Optional[str] = Field(
        "float", description="The format of the embeddings"
    )
    user: Optional[str] = None
    dimensions: Optional[int] = None

    # Allow any additional fields
    class Config:
        extra = "allow"  # This allows additional fields not defined in the model

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields = {"model", "input", "encoding_format", "user", "dimensions"}
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
