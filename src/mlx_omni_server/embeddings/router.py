from fastapi import APIRouter, HTTPException

from .embeddings_service import EmbeddingsService
from .schema import EmbeddingRequest, EmbeddingResponse

router = APIRouter(tags=["embeddings"])
embeddings_service = EmbeddingsService()


@router.post("/embeddings", response_model=EmbeddingResponse)
@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for text input.

    This endpoint generates vector representations of input text,
    which can be used for semantic search, clustering, and other NLP tasks.
    """
    try:
        return embeddings_service.generate_embeddings(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
