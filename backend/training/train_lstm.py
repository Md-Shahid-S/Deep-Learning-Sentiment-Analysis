import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from app.models.lstm_model import build_lstm_model
from app.services.preprocessor import NLTKPreprocessor
from app.core.config import settings
import json

def load_data():
    print("Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    def decode_review(text):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])
    
    X_train_text = [decode_review(x) for x in x_train]
    X_test_text = [decode_review(x) for x in x_test]
    
    return X_train_text, X_test_text, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    preprocessor = NLTKPreprocessor()
    print("Preprocessing data...")
    X_train_clean = [preprocessor.process(t) for t in X_train[:10000]]
    X_test_clean = [preprocessor.process(t) for t in X_test[:2000]]
    
    # Tokenization
    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_clean)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train_clean)
    X_test_seq = tokenizer.texts_to_sequences(X_test_clean)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=settings.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=settings.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Build model
    print("Building LSTM model...")
    model = build_lstm_model(
        vocab_size=10000,
        embed_dim=settings.EMBEDDING_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        dropout=settings.DROPOUT
    )
    
    # Train
    print("Training...")
    model.fit(
        X_train_pad, y_train[:10000],
        epochs=5,
        batch_size=32,
        validation_data=(X_test_pad, y_test[:2000]),
        verbose=1
    )
    
    # Save model and tokenizer
    print("Saving artifacts...")
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(settings.LSTM_MODEL_PATH)
    
    tokenizer_json = tokenizer.to_json()
    with open(settings.TOKENIZER_PATH, 'w') as f:
        f.write(tokenizer_json)
    
    print("LSTM Training Complete.")

if __name__ == "__main__":
    main()
