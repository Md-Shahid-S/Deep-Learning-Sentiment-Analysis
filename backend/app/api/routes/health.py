from fastapi import APIRouter
from app.core.model_loader import models

router = APIRouter()

@router.get("/health")
async def health_check():
    loaded_models = {k: v is not None for k, v in models.items()}
    return {
        "status": "healthy",
        "models_status": loaded_models
    }
