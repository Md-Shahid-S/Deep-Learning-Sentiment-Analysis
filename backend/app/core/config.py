import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    PROJECT_NAME: str = "Sentiment Analysis API"
    VERSION: str = "1.0.0"
    
    # Paths
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    DATA_DIR: Path = BASE_DIR.parent / "data"
    
    # Model Paths
    LSTM_MODEL_PATH: Path = ARTIFACTS_DIR / "lstm_model.h5"
    TOKENIZER_PATH: Path = ARTIFACTS_DIR / "tokenizer.json"
    SVM_MODEL_PATH: Path = ARTIFACTS_DIR / "svm_model.joblib"
    LOGREG_MODEL_PATH: Path = ARTIFACTS_DIR / "logreg_model.joblib"
    TFIDF_VECTORIZER_PATH: Path = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    
    # LSTM Config
    MAX_SEQUENCE_LENGTH: int = 100
    EMBEDDING_DIM: int = 100
    HIDDEN_DIM: int = 64
    DROPOUT: float = 0.2
    
    # API Config
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()

# Ensure artifacts directory exists
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
