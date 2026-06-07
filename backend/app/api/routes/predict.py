from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictRequest, PredictResponse
from app.services.predictor import predict_sentiment

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        return predict_sentiment(request.text, request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch")
async def predict_batch(texts: list[PredictRequest]):
    return [predict_sentiment(r.text, r.model) for r in texts]