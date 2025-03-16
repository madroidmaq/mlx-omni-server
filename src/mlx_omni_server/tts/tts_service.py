from pathlib import Path

from typing import Mapping, Type
from typing_extensions import override

from f5_tts_mlx.generate import generate
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model

from .schema import TTSRequest

from pydantic import BaseModel, Field#, PrivateAttr

import mlx.nn as nn
import numpy as np
import soundfile as sf


AUDIO_MODEL_MAP: Mapping[str, 'Type[TTSModelAdapter]'] = {
    "lucasnewman/f5-tts-mlx": 'F5Model',
    "prince-canuma/Kokoro-82M": 'KokoroModel',
    "mlx-community/Kokoro-82M-bf16": 'KokoroModel',
    "mlx-community/Kokoro-82M-8bit": 'KokoroModel',
    "mlx-community/Kokoro-82M-6bit": 'KokoroModel',
    "mlx-community/Kokoro-82M-4bit": 'KokoroModel',
}
# Note: `KokoroModel` requires `path_or_hf_repo` field to be set
# so this fallback solution needs to be updated if the current
# default model adapter `F5Model` is changed.
FALLBACK_MODEL_ADAPTER: 'Type[TTSModelAdapter]' = 'F5Model'

class TTSModelAdapter(BaseModel):
    """Base class to adapt different TTS models to support the audio endpoint."""
    path_or_hf_repo: str | Path = Field(None, title="The path or the huggingface repository to load the model from.")

    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        """
        Generate audio from input text.
        
        Args:
            request (TTSRequest): The request object containing the input text and other parameters.
            output_path (str | Path): The path to save the generated audio file.
            
        Returns:
            bool: True if the audio was generated successfully, False otherwise.
        """
        pass

    @classmethod
    def from_path_or_hf_repo(cls, path_or_hf_repo: str) -> 'TTSModelAdapter':
        model_class = AUDIO_MODEL_MAP.get(path_or_hf_repo)
        if model_class is None:
            if not FALLBACK_MODEL_ADAPTER:
                raise ValueError(f"Model {path_or_hf_repo} not found in model map")
            model_class = FALLBACK_MODEL_ADAPTER
        return globals()[model_class](path_or_hf_repo=path_or_hf_repo)


class F5Model(TTSModelAdapter):

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        generate(
            model_name=request.model,
            generation_text=request.input,
            speed=request.speed,
            output_path=str(output_path),
            **(request.get_extra_params() or {}),
        )
        return Path(output_path).exists()



class KokoroModel(TTSModelAdapter):

    path_or_hf_repo: str = Field("prince-canuma/Kokoro-82M", description="The Kokoro MLX converted model path or huggingface repository to load from.")
    voice: str = Field("af_sky", description="The Kokoro voice to use when generating speech.")
    # TODO: Eval performance of using PrivateAttr or `functools.cached_property` for these properties
    # _model: nn.Module | None = PrivateAttr(None, "The Kokoro model to use when generating speech.")
    # _pipeline: KokoroPipeline | None = PrivateAttr(None, "A pipeline to render the voice with.")

    @property
    def lang_code(self):
        """
        Language codes:
        🇺🇸 'a' => American English, 🇬🇧 'b' => British English
        🇯🇵 'j' => Japanese: pip install misaki[ja]
        🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
        🇪🇸 'e' => Spanish es
        🇫🇷 'f' => French fr-fr
        🇮🇳 'h' => Hindi hi
        🇮🇹 'i' => Italian it
        🇧🇷 'p' => Brazilian Portuguese pt-br
        """
        return self.voice[:1]
    
    @property
    def voice_gender(self):
        """
        Voice gender:
        👨 'm' => 'male'
        👩 'f' => 'female'
        """
        return self.voice[1:2]
    
    @property
    def model(self) -> nn.Module:
        """
        The Kokoro model used when generating speech.
        """
        return load_model(self.path_or_hf_repo)
    
    @property
    def pipeline(self):
        pipeline = KokoroPipeline(
            lang_code=self.lang_code,
            model=self.model,
            repo_id=self.path_or_hf_repo,
        )
        return pipeline

    @override
    def generate_audio(self, request, output_path):
        if request.voice:
            self.voice = request.voice

        generator = self.pipeline(
            text=request.input,
            voice=self.voice,
            speed=request.speed,
            split_pattern=f'\n+',  # Split on newlines
        )

        audio_chunks = []

        for i, (gs, ps, audio) in enumerate(generator):
            IS_KOKORO_MLX_MODEL = True # MLX Kokoro models return tuple
            audio = audio[0] if IS_KOKORO_MLX_MODEL else audio
            audio_chunks.append(audio)
        
        audio = np.concatenate(audio_chunks)
        sf.write(output_path, audio, 24000)
        return Path(output_path).exists()


class TTSService:
    model: TTSModelAdapter

    def __init__(self, path_or_hf_repo: str | Path | None = None):
        self.model = TTSModelAdapter.from_path_or_hf_repo(path_or_hf_repo)
        self.sample_audio_path = Path("sample.wav")

    async def generate_speech(
        self,
        request: TTSRequest,
    ) -> bytes:
        try:
            self.model.generate_audio(
                request=request, output_path=self.sample_audio_path
            )
            with open(self.sample_audio_path, "rb") as audio_file:
                audio_content = audio_file.read()
            self.sample_audio_path.unlink(missing_ok=True)
            return audio_content
        except Exception as e:
            raise Exception(f"Error reading audio file: {str(e)}")
