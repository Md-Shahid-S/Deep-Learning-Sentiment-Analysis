import streamlit as st
import pandas as pd
import requests
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("📊 Model Benchmarking")
st.write("Compare the performance and speed of different models on the same input.")

test_review = st.text_area("Enter a test review:", "I absolutely loved this movie! The acting was superb and the plot was gripping.", height=100)

if st.button("Run Benchmark"):
    results = []
    models = ["lstm", "svm", "logistic_regression"]
    
    progress_bar = st.progress(0)
    
    for i, model_name in enumerate(models):
        start_time = time.time()
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": test_review, "model": model_name})
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "Model": model_name.upper(),
                    "Sentiment": data["sentiment"],
                    "Confidence": f"{data['confidence']*100:.2f}%",
                    "Latency (ms)": round((end_time - start_time) * 1000, 2)
                })
            else:
                results.append({"Model": model_name.upper(), "Sentiment": "Error", "Confidence": "N/A", "Latency (ms)": "N/A"})
        except Exception as e:
            results.append({"Model": model_name.upper(), "Sentiment": "Failed", "Confidence": "N/A", "Latency (ms)": "N/A"})
        
        progress_bar.progress((i + 1) / len(models))

    st.write("### Comparison Results")
    comparison_df = pd.DataFrame(results)
    st.table(comparison_df)
    
    st.write("### Model Architecture Details")
    st.info("""
    - **LSTM**: Bidirectional Long Short-Term Memory network. Good for capturing long-range dependencies and context.
    - **SVM**: Support Vector Machine with linear kernel. High performance on high-dimensional text data.
    - **Logistic Regression**: Reliable baseline for binary classification tasks.
    """)
    
    # Speed comparison chart
    if "Latency (ms)" in comparison_df.columns and comparison_df["Latency (ms)"].dtype != object:
        st.write("### Latency Comparison (Lower is better)")
        st.bar_chart(comparison_df.set_index("Model")["Latency (ms)"])
