from typing import Optional

from fastapi import APIRouter, Query

from .models_service import AnthropicModelsService
from .schema import AnthropicModelList

router = APIRouter(tags=["anthropic"])
models_service = AnthropicModelsService()


@router.get("/models", response_model=AnthropicModelList)
@router.get("/v1/models", response_model=AnthropicModelList)
async def list_anthropic_models(
    before_id: Optional[str] = Query(
        default=None,
        title="Before Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately before this object.",
    ),
    after_id: Optional[str] = Query(
        default=None,
        title="After Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately after this object.",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=1000,
        title="Limit",
        description="Number of items to return per page. Defaults to 20. Ranges from 1 to 1000.",
    ),
) -> AnthropicModelList:
    """List available models in Anthropic format."""
    return models_service.list_models(
        limit=limit, after_id=after_id, before_id=before_id
    )
