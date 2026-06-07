import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

def build_lstm_model(vocab_size, embed_dim, hidden_dim, dropout, pretrained_embeddings=None):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
            weights=[pretrained_embeddings] if pretrained_embeddings is not None else None,
            trainable=True if pretrained_embeddings is None else False
        ),
        Bidirectional(LSTM(hidden_dim, dropout=dropout, recurrent_dropout=0)),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
