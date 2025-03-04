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
    input: str = Field(..., max_length=4096)
    voice: str = Field(
        ..., description="Voice to use (e.g. alloy, echo, fable, onyx, nova, shimmer)"
    )
    response_format: Optional[AudioFormat] = Field(default=AudioFormat.WAV)
    speed: Optional[float] = Field(default=1.0)

    # Allow any additional fields
    class Config:
        extra = "allow"  # This allows additional fields not defined in the model

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard OpenAI API."""
        standard_fields = {"model", "input",
                           "voice", "response_format", "speed"}
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}

    @field_validator("speed")
    def validate_speed(cls, v):
        if v < 0.25 or v > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        return v

    @field_validator("model")
    def validate_model(cls, v):
        valid_models = [
            "lucasnewman/f5-tts-mlx",
            "mlx-community/Kokoro-82M-bf16",
            "mlx-community/Kokoro-82M-4bit",
            "mlx-community/Kokoro-82M-6bit",
            "mlx-community/Kokoro-82M-8bit"
        ]
        if not any(v.startswith(model) for model in valid_models):
            raise ValueError(f'Model must be one of: {
                             ", ".join(valid_models)}')
        return v
