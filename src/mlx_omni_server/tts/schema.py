from enum import Enum
from typing import Any, Dict, Optional, TypeAlias, Literal, Union, List

from pydantic import BaseModel, Field, field_validator

KokoroVoice: TypeAlias = Literal["am_santa", "af_sarah", "bf_isabella", "af_sky", "af_river", "jm_kumo", "af_kore", "zf_xiaoyi", "zf_xiaoni", "am_adam", "bf_alice", "am_michael", "af_jessica", "jf_nezumi", "bf_emma", "jf_tebukuro", "af_nova", "jf_alpha", "bf_lily", "zf_xiaobei", "am_fenrir", "am_onyx", "bm_daniel", "bm_fable", "am_liam", "jf_gongitsune", "af_nicole", "am_puck", "af_alloy", "af_aoede", "zf_xiaoxiao", "af_heart", "af_bella"]

# Temporary fix to allow long text, maybe use another map?
RAISE_ON_INVALID_INPUT_LONG_TEXT = False

VALID_MODELS = [
    "lucasnewman/f5-tts-mlx",
    "prince-canuma/Kokoro-82M",
    "mlx-community/Kokoro-82M-bf16",
    "mlx-community/Kokoro-82M-8bit",
    "mlx-community/Kokoro-82M-6bit",
    "mlx-community/Kokoro-82M-4bit",
]

# These OpenAI voices were mentioned in the TTSRequest.voice field so I added them here
alt_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
kokoro_voices: List[KokoroVoice] = ["am_santa", "af_sarah", "bf_isabella", "af_sky", "af_river", "jm_kumo", "af_kore", "zf_xiaoyi", "zf_xiaoni", "am_adam", "bf_alice", "am_michael", "af_jessica", "jf_nezumi", "bf_emma", "jf_tebukuro", "af_nova", "jf_alpha", "bf_lily", "zf_xiaobei", "am_fenrir", "am_onyx", "bm_daniel", "bm_fable", "am_liam", "jf_gongitsune", "af_nicole", "am_puck", "af_alloy", "af_aoede", "zf_xiaoxiao", "af_heart", "af_bella"]


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
    voice: Union[str, KokoroVoice] = Field(
        default="af_sky",
        description="Voice used, choose correct voice for selected model.",
        examples=kokoro_voices + alt_voices
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

    @field_validator("input")
    def validate_input(cls, v):
        if len(v) > 4096 and RAISE_ON_INVALID_INPUT_LONG_TEXT:
            raise ValueError("Input text must be less than 4096 characters")
        return v

    @field_validator("speed")
    def validate_speed(cls, v):
        if v < 0.25 or v > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        return v

    @field_validator("model")
    def validate_model(cls, v):
        if v not in VALID_MODELS:
            raise ValueError(f'Model must be one of: {", ".join(VALID_MODELS)}')
        return v
