import numpy as np
import tensorflow as tf
from app.models.baseline_models import train_logistic_regression, train_svm, save_baseline_model
from app.services.preprocessor import NLTKPreprocessor
from app.core.config import settings
from sklearn.metrics import accuracy_score

def load_data():
    print("Loading IMDB dataset...")
    # Load IMDB dataset from keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    
    # Get word index to convert back to text for baselines
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
    X_train_clean = [preprocessor.process(t) for t in X_train[:5000]] # Limit for speed
    y_train_subset = y_train[:5000]
    
    X_test_clean = [preprocessor.process(t) for t in X_test[:1000]]
    y_test_subset = y_test[:1000]

    # Train Logistic Regression
    print("Training Logistic Regression...")
    tfidf_lr, lr_model = train_logistic_regression(X_train_clean, y_train_subset)
    y_pred_lr = lr_model.predict(tfidf_lr.transform(X_test_clean))
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test_subset, y_pred_lr):.4f}")
    save_baseline_model(tfidf_lr, lr_model, "logreg", settings.ARTIFACTS_DIR)

    # Train SVM
    print("Training SVM...")
    tfidf_svm, svm_model = train_svm(X_train_clean, y_train_subset)
    y_pred_svm = svm_model.predict(tfidf_svm.transform(X_test_clean))
    print(f"SVM Accuracy: {accuracy_score(y_test_subset, y_pred_svm):.4f}")
    save_baseline_model(tfidf_svm, svm_model, "svm", settings.ARTIFACTS_DIR)

if __name__ == "__main__":
    main()
