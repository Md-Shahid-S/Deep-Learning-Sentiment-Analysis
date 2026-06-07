from pydantic import BaseModel
from typing import Literal

class PredictRequest(BaseModel):
    text: str
    model: Literal["lstm", "svm", "logistic_regression"] = "lstm"

class PredictResponse(BaseModel):
    sentiment: Literal["positive", "negative"]
    confidence: float
    model_used: str
    preprocessed_text: str
    tokens_count: int