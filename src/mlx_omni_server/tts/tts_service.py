from .schema import TTSRequest
from f5_tts_mlx.generate import generate
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro import KokoroPipeline
from pathlib import Path
import soundfile as sf


class F5Model:

    def __init__(self):
        pass

    def generate_audio(self, request: TTSRequest, output_path):
        generate(
            model_name=request.model,
            generation_text=request.input,
            speed=request.speed,
            output_path=output_path,
            **(request.get_extra_params() or {}),
        )


class KokoroModel:
    def __init__(self):
        self.models = {}

    def generate_audio(self, request: TTSRequest, output_path):
        # Load model if not already loaded
        if request.model not in self.models:
            model = load_model(request.model)
            self.models[request.model] = KokoroPipeline(
                lang_code=request.lang_code,  # This should come from request params
                model=model,
                repo_id=request.model
            )

        pipeline = self.models[request.model]

        # Generate audio using pipeline
        for _, _, audio in pipeline(
            request.input,
            voice=request.get_extra_params().get('voice', request.voice),
            speed=request.speed
        ):
            # Save the generated audio
            sf.write(output_path, audio[0], 24000)


class TTSService:
    def __init__(self):
        self.f5_model = F5Model()
        self.mlx_model = KokoroModel()
        self.sample_audio_path = Path("sample.wav")

    async def generate_speech(
        self,
        request: TTSRequest,
    ) -> bytes:
        try:
            # Determine which model to use based on the model name
            if request.model.startswith("mlx-community/"):
                print("Using Kokoro model")
                self.mlx_model.generate_audio(
                    request=request, output_path=self.sample_audio_path
                )
            else:
                self.f5_model.generate_audio(
                    request=request, output_path=self.sample_audio_path
                )

            with open(self.sample_audio_path, "rb") as audio_file:
                audio_content = audio_file.read()
            self.sample_audio_path.unlink(missing_ok=True)
            return audio_content
        except Exception as e:
            raise Exception(f"Error reading audio file: {str(e)}")
