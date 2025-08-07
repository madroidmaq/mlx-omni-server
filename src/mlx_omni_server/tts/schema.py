from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class AudioFormat(str, Enum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class TTSRequest(BaseModel):
    model: str = Field(..., description="TTS model to use")
    input: str = Field(...)
    voice: str = Field(
        default="af_sky",
        description="Voice used, choose correct voice for selected model.",
    )
    response_format: Optional[AudioFormat] = Field(default=AudioFormat.WAV)
    speed: Optional[float] = Field(default=1.0)

    # Allow any additional fields
    class Config:
        extra = "allow"  # This allows additional fields not defined in the model

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields = {"model", "input", "voice", "response_format", "speed"}
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}

    @field_validator("speed")
    def validate_speed(cls, v):
        if v < 0.25 or v > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        return v
