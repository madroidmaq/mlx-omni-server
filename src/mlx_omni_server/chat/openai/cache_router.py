from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mlx_omni_server.chat.mlx.wrapper_cache import wrapper_cache

router = APIRouter(tags=["cache"])


@router.get("/v1/cache")
async def get_cache_info():
    """Return current ChatGenerator cache information."""
    return JSONResponse(content=wrapper_cache.get_cache_info())


@router.post("/v1/cache/clear")
async def clear_cache():
    """Clear cached ChatGenerator instances."""
    cleared_count = wrapper_cache.clear_cache()
    return JSONResponse(
        content={
            "cleared_count": cleared_count,
            "cache_info": wrapper_cache.get_cache_info(),
        }
    )
