# 🎬 Sentiment Analysis Project

A full-stack Sentiment Analysis application featuring a **FastAPI** backend and a **Streamlit** frontend. The project implements three different machine learning models to classify movie reviews as Positive or Negative.

## 🚀 Features
- **Multiple Models:** Compare results from Bidirectional LSTM (TensorFlow), SVM, and Logistic Regression.
- **Real-time Inference:** Single review analysis with confidence scores.
- **Batch Processing:** Upload a CSV file for bulk sentiment analysis.
- **Interactive UI:** Visualization of results using Plotly.
- **Containerized:** Easy deployment using Docker and Docker Compose.

## 🛠️ Tech Stack
- **Backend:** FastAPI, TensorFlow/Keras, Scikit-learn, NLTK.
- **Frontend:** Streamlit, Plotly, Pandas.
- **DevOps:** Docker, Docker Compose.
- **NLP:** NLTK (Lemmatization, Tokenization), TF-IDF, Word Embeddings.

## 📁 Project Structure
```text
.
├── backend/               # FastAPI Application
│   ├── app/               # Core Logic & API Routes
│   ├── artifacts/         # Saved Models & Tokenizers
│   ├── training/          # Model Training Scripts
│   └── Dockerfile         # Backend Container Definition
├── frontend/              # Streamlit Application
│   ├── components/        # Reusable UI & Charts
│   ├── pages/             # Multi-page Application Logic
│   └── Dockerfile         # Frontend Container Definition
├── data/                  # Data Placeholders
├── docker-compose.yaml    # Multi-container Orchestration
└── .env                   # Environment Variables
```

## ⚙️ Setup & Execution
Please refer to [Exception.md](./Exception.md) (Execution Guide) for detailed instructions on how to train the models and run the application.

## 🧠 Models
1. **LSTM (Deep Learning):** A bidirectional LSTM network built with TensorFlow. It captures context from both directions of the text sequence.
2. **SVM (Baseline):** Support Vector Machine with a linear kernel, utilizing TF-IDF vectorization.
3. **Logistic Regression:** A classic statistical model for binary classification.

## 📊 Dataset
The models are trained on the **IMDB Movie Reviews Dataset** (50,000 reviews), known for its balance and high-quality labels.
