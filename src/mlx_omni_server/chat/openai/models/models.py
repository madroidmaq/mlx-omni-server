from fastapi import APIRouter, HTTPException, Request

from .models_service import ModelsService
from .schema import Model, ModelDeletion, ModelList

router = APIRouter(tags=["models"])

# Lazy initialization to avoid scanning cache during module import
_models_service = None


def get_models_service() -> ModelsService:
    """Get or create the models service singleton with lazy initialization."""
    global _models_service
    if _models_service is None:
        _models_service = ModelsService()
    return _models_service


def extract_model_id_from_path(request: Request) -> str:
    """Extract full model ID from request path"""
    path = request.url.path
    prefix = "/v1/models/" if "/v1/models/" in path else "/models/"
    return path[len(prefix) :]


def handle_model_error(e: Exception) -> None:
    """Handle model-related errors and raise appropriate HTTP exceptions"""
    if isinstance(e, ValueError):
        raise HTTPException(status_code=404, detail=str(e))
    print(f"Error processing request: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelList)
@router.get("/v1/models", response_model=ModelList)
async def list_models(include_details: bool = False) -> ModelList:
    """List all available models"""
    return get_models_service().list_models(include_details)


@router.get("/models/{model_id:path}", response_model=Model)
@router.get("/v1/models/{model_id:path}", response_model=Model)
async def get_model(model_id: str, include_details: bool = False) -> Model:
    """Get information about a specific model"""
    model = get_models_service().get_model(model_id, include_details)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/models/{model_id:path}", response_model=ModelDeletion)
@router.delete("/v1/models/{model_id:path}", response_model=ModelDeletion)
async def delete_model(request: Request) -> ModelDeletion:
    """
    Delete a fine-tuned model from local cache.
    """
    try:
        model_id = extract_model_id_from_path(request)
        return get_models_service().delete_model(model_id)
    except Exception as e:
        handle_model_error(e)
