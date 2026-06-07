import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from app.core.model_loader import get_model
from app.services.preprocessor import NLTKPreprocessor
from app.core.config import settings
from app.models.schemas import PredictResponse

preprocessor = NLTKPreprocessor()

def predict_sentiment(text: str, model_name: str) -> PredictResponse:
    # 1. Preprocess
    clean_text = preprocessor.process(text)
    tokens = preprocessor.tokenize(text)
    
    model = get_model(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} is not loaded. Please train it first.")

    if model_name == "lstm":
        tokenizer = get_model("tokenizer")
        if tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        
        # LSTM prediction
        sequences = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(sequences, maxlen=settings.MAX_SEQUENCE_LENGTH)
        
        prediction = model.predict(padded)[0][0]
        sentiment = "positive" if prediction >= 0.5 else "negative"
        confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
        
    else:
        # SVM or Logistic Regression prediction
        tfidf = get_model("tfidf")
        if tfidf is None:
            raise ValueError("TF-IDF vectorizer not loaded.")
            
        vectorized_text = tfidf.transform([clean_text])
        prediction_prob = model.predict_proba(vectorized_text)[0]
        
        # scikit-learn classes_ usually [0, 1] mapping to [negative, positive]
        pred_idx = np.argmax(prediction_prob)
        sentiment = "positive" if pred_idx == 1 else "negative"
        confidence = float(prediction_prob[pred_idx])

    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        model_used=model_name,
        preprocessed_text=clean_text,
        tokens_count=len(tokens)
    )
