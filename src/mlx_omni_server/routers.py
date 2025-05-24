from fastapi import APIRouter

from .chat import router as chat_router
from .chat.models import models
from .embeddings import router as embeddings_router
from .images import images
from .stt import stt as stt_router
from .tts import tts as tts_router

api_router = APIRouter()
api_router.include_router(stt_router.router)
api_router.include_router(tts_router.router)
api_router.include_router(models.router)
api_router.include_router(images.router)
api_router.include_router(chat_router.router)
api_router.include_router(embeddings_router.router)
