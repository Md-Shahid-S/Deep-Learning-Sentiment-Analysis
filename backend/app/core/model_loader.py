import joblib
import tensorflow as tf
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Global variables to store models
models = {
    "lstm": None,
    "tokenizer": None,
    "svm": None,
    "logistic_regression": None,
    "tfidf": None
}

def load_all_models():
    """Load all models into memory at startup."""
    try:
        # Load LSTM
        if settings.LSTM_MODEL_PATH.exists():
            models["lstm"] = tf.keras.models.load_model(settings.LSTM_MODEL_PATH)
            logger.info("LSTM model loaded.")
        
        if settings.TOKENIZER_PATH.exists():
            with open(settings.TOKENIZER_PATH, 'r') as f:
                json_string = f.read()
                models["tokenizer"] = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
            logger.info("Tokenizer loaded.")

        # Load Baselines
        if settings.SVM_MODEL_PATH.exists():
            models["svm"] = joblib.load(settings.SVM_MODEL_PATH)
            logger.info("SVM model loaded.")
            
        if settings.LOGREG_MODEL_PATH.exists():
            models["logistic_regression"] = joblib.load(settings.LOGREG_MODEL_PATH)
            logger.info("Logistic Regression model loaded.")
            
        if settings.TFIDF_VECTORIZER_PATH.exists():
            models["tfidf"] = joblib.load(settings.TFIDF_VECTORIZER_PATH)
            logger.info("TF-IDF vectorizer loaded.")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def get_model(name: str):
    return models.get(name)
