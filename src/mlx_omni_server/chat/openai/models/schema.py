from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_serializer


class Model(BaseModel):
    """Model information as per OpenAI API specification"""

    id: str = Field(..., description="The model identifier")
    object: str = Field(default="model", description="The object type (always 'model')")
    created: int = Field(
        ..., description="Unix timestamp of when the model was created"
    )
    owned_by: str = Field(..., description="Organization that owns the model")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Full model configuration (if details are requested)"
    )

    @model_serializer
    def serialize_model(self):
        """Custom serializer to exclude None details field"""
        data = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
        }
        if self.details is not None:
            data["details"] = self.details
        return data


class ModelList(BaseModel):
    """Response format for list of models"""

    object: str = Field(default="list", description="The object type (always 'list')")
    data: List[Model] = Field(..., description="List of model objects")


class ModelDeletion(BaseModel):
    """Response format for model deletion"""

    id: str = Field(..., description="The ID of the deleted model")
    object: str = Field(default="model", description="The object type (always 'model')")
    deleted: bool = Field(..., description="Whether the model was deleted")
