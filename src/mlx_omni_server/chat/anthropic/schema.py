from typing import List, Optional

from pydantic import BaseModel, Field


class AnthropicModelInfo(BaseModel):
    """Model information as per Anthropic API specification"""

    id: str = Field(..., description="Unique model identifier.")
    display_name: str = Field(..., description="A human-readable name for the model.")
    created_at: str = Field(
        ...,
        description="RFC 3339 datetime string representing the time at which the model was released.",
    )
    type: str = Field(
        default="model",
        description='Object type. For Models, this is always "model".',
    )


class AnthropicModelList(BaseModel):
    """Response format for list of models, Anthropic style."""

    data: List[AnthropicModelInfo] = Field(..., description="List of model objects")
    first_id: Optional[str] = Field(
        None,
        description="First ID in the `data` list. Can be used as the `before_id` for the previous page.",
    )
    last_id: Optional[str] = Field(
        None,
        description="Last ID in the `data` list. Can be used as the `after_id` for the next page.",
    )
    has_more: bool = Field(
        ...,
        description="Indicates if there are more results in the requested page direction.",
    )
