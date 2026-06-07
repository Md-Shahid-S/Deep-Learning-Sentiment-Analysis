import streamlit as st

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Deep Learning Sentiment Analysis")
st.caption("LSTM · GloVe · SVM · Logistic Regression | Trained on 50K IMDb Reviews")

st.sidebar.title("Navigation")
st.sidebar.info("Use the pages below to explore predictions and benchmarks.")