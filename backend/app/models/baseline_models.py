from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from pathlib import Path

def train_logistic_regression(X_train, y_train):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    return tfidf, model

def train_svm(X_train, y_train):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Use probability=True so we can get confidence scores
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_tfidf, y_train)
    
    return tfidf, model

def save_baseline_model(tfidf, model, model_name, artifacts_dir: Path):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, artifacts_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, artifacts_dir / f"{model_name}_model.joblib")
