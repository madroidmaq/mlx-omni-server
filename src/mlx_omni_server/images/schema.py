from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ImageSize(str, Enum):
    S256x256 = "256x256"
    S512x512 = "512x512"
    S1024x1024 = "1024x1024"
    S1792x1024 = "1792x1024"
    S1024x1792 = "1024x1792"


class ImageQuality(str, Enum):
    STANDARD = "standard"
    HD = "hd"


class ImageStyle(str, Enum):
    VIVID = "vivid"
    NATURAL = "natural"


class ResponseFormat(str, Enum):
    URL = "url"
    B64_JSON = "b64_json"


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., max_length=4000)
    model: Optional[str] = Field(
        default="dhairyashil/FLUX.1-schnell-mflux-4bit",
        description="The model to use for image generation",
    )
    n: Optional[int] = Field(default=1, ge=1, le=10)
    quality: Optional[ImageQuality] = Field(default=ImageQuality.STANDARD)
    response_format: Optional[ResponseFormat] = Field(default=ResponseFormat.B64_JSON)
    size: Optional[ImageSize] = Field(default=ImageSize.S1024x1024)
    style: Optional[ImageStyle] = Field(default=ImageStyle.VIVID)
    user: Optional[str] = None

    # Allow any additional fields
    class Config:
        extra = "allow"  # This allows additional fields not defined in the model

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields = {
            "prompt",
            "model",
            "n",
            "quality",
            "response_format",
            "size",
            "style",
            "user",
        }
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}

    @field_validator("prompt")
    def validate_prompt_length(cls, v, values):
        max_length = 4000
        if len(v) > max_length:
            raise ValueError(
                f"Prompt length exceeds maximum of {max_length} characters"
            )
        return v


class ImageObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageObject]
