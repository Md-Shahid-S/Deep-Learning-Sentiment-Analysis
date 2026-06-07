import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from app.core.config import settings
from app.core.model_loader import load_all_models, get_model
from app.services.preprocessor import NLTKPreprocessor
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_test_data():
    print("Loading IMDB test data...")
    (_, _), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    X_test_text = [decode_review(x) for x in x_test[:1000]]
    return X_test_text, y_test[:1000]

def evaluate_all():
    X_test_text, y_test = load_test_data()
    preprocessor = NLTKPreprocessor()
    X_test_clean = [preprocessor.process(t) for t in X_test_text]
    
    load_all_models()
    
    models_to_eval = ["lstm", "svm", "logistic_regression"]
    
    for name in models_to_eval:
        print(f"\n--- Evaluating {name.upper()} ---")
        model = get_model(name)
        if model is None:
            print(f"Model {name} not found. Skipping.")
            continue
            
        if name == "lstm":
            tokenizer = get_model("tokenizer")
            X_seq = tokenizer.texts_to_sequences(X_test_clean)
            X_pad = pad_sequences(X_seq, maxlen=settings.MAX_SEQUENCE_LENGTH)
            y_pred_prob = model.predict(X_pad)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            tfidf = get_model("tfidf")
            X_tfidf = tfidf.transform(X_test_clean)
            y_pred = model.predict(X_tfidf)
            
        print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

if __name__ == "__main__":
    evaluate_all()
