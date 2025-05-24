from pathlib import Path

from f5_tts_mlx.generate import generate
from mlx_audio.tts.generate import generate_audio
from pydantic import BaseModel, Field  # , PrivateAttr
from typing_extensions import override

from .schema import TTSRequest


class TTSModelAdapter(BaseModel):
    """Base class to adapt different TTS models to support the audio endpoint."""

    path_or_hf_repo: str | Path = Field(
        None, title="The path or the huggingface repository to load the model from."
    )

    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        """
        Generate audio from input text.
        ¨
        Args:
            request (TTSRequest): The request object containing the input text and other parameters.
            output_path (str | Path): The path to save the generated audio file.

        Returns:
            bool: True if the audio was generated successfully, False otherwise.
        """
        pass

    @classmethod
    def from_path_or_hf_repo(cls, path_or_hf_repo: str) -> "TTSModelAdapter":
        if path_or_hf_repo == "lucasnewman/f5-tts-mlx":
            return F5Model(path_or_hf_repo=path_or_hf_repo)
        else:
            return MlxAudioModel(path_or_hf_repo=path_or_hf_repo)


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


class MlxAudioModel(TTSModelAdapter):
    path_or_hf_repo: str = Field("mlx-community/Kokoro-82M-4bit")

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        voice = request.voice if hasattr(request, "voice") else "af_sky"
        lang_code = voice[:1]

        extra_params = request.get_extra_params() or {}

        generate_audio(
            text=request.input,
            model_path=self.path_or_hf_repo,
            voice=voice,
            speed=request.speed,
            lang_code=lang_code,
            file_prefix=str(output_path).rsplit(".", 1)[0],
            audio_format=request.response_format.value,
            sample_rate=24000,
            join_audio=True,
            verbose=False,
            **extra_params,
        )

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
