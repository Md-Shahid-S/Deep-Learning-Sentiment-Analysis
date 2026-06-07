import streamlit as st
import requests, os

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("🔍 Single Review Analysis")

model_choice = st.selectbox(
    "Choose Model",
    ["lstm", "svm", "logistic_regression"],
    format_func=lambda x: {
        "lstm": "🧠 Bidirectional LSTM + GloVe (91%)",
        "svm": "⚡ SVM + TF-IDF (89%)",
        "logistic_regression": "📊 Logistic Regression + TF-IDF (88%)"
    }[x]
)

review = st.text_area("Paste your movie review:", height=200,
    placeholder="This film was absolutely brilliant...")

if st.button("Analyze Sentiment", type="primary"):
    with st.spinner("Running inference..."):
        resp = requests.post(f"{API_URL}/predict",
            json={"text": review, "model": model_choice})
        result = resp.json()

    col1, col2 = st.columns(2)
    sentiment = result["sentiment"]
    confidence = result["confidence"]

    with col1:
        color = "🟢" if sentiment == "positive" else "🔴"
        st.metric("Sentiment", f"{color} {sentiment.upper()}")
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.metric("Model", result["model_used"])

    with col2:
        st.progress(confidence)
        st.caption(f"Preprocessed to {result['tokens_count']} tokens")
        with st.expander("Show preprocessed text"):
            st.code(result["preprocessed_text"])