from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import predict, health
from app.core.model_loader import load_all_models
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()       # loads all 3 models once at startup
    yield

app = FastAPI(title="Sentiment Analysis API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.include_router(predict.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")