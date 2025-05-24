import logging
import os

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from src.mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


def test_speech(openai_client):
    """Test text-to-speech functionality using OpenAI client"""
    try:
        model = "lucasnewman/f5-tts-mlx"
        response = openai_client.audio.speech.create(
            model=model,
            input="The quick brown fox jumped over the lazy dog.",
            voice="alloy",
        )
        logger.info(f"Speech Response: {response}")

        # Validate response
        assert response is not None, "No response received"
        # If further validation of audio content is needed, add more assertions

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_mlx_audio_kokoro_speech(openai_client):
    """Test text-to-speech functionality using OpenAI client"""
    try:
        model = "mlx-community/Kokoro-82M-4bit"
        response = openai_client.audio.speech.create(
            model=model,
            input="The quick brown fox jumped over the lazy dog.",
            voice="af_sky",
        )
        logger.info(f"MLX Audio Speech Response: {response}")

        # Validate response
        assert response is not None, "No response received"
        # If further validation of audio content is needed, add more assertions

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_mlx_audio_dia_speech(openai_client):
    """Test text-to-speech functionality using OpenAI client"""
    try:
        model = "mlx-community/Dia-1.6B-4bit"
        response = openai_client.audio.speech.create(
            model=model,
            input="The quick brown fox jumped over the lazy dog.",
            voice="demo",
        )
        logger.info(f"MLX Audio Speech Response: {response}")

        # Validate response
        assert response is not None, "No response received"
        # If further validation of audio content is needed, add more assertions

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise


def test_transcription(openai_client):
    """Test audio transcription functionality using OpenAI client"""
    try:
        audio_file_path = "tests/test_audio.wav"

        if not os.path.exists(audio_file_path):
            pytest.skip(f"Audio file {audio_file_path} does not exist")

        model = "mlx-community/whisper-large-v3-turbo"

        with open(audio_file_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )

        logger.info(f"Transcription response: {response}")

        # Validate response
        assert response is not None, "No response received"
        assert hasattr(response, "text"), "Response does not contain text field"
        assert "MLX" in response.text, "Transcription text is empty"

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise
