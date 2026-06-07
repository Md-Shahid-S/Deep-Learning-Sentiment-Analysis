import streamlit as st
import pandas as pd
import requests
import os
from components.charts import plot_sentiment_distribution, plot_confidence_histogram

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.title("📂 Batch Analysis")
st.write("Upload a CSV file with a column named 'text' to analyze multiple reviews at once.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        st.write(f"Loaded {len(df)} reviews.")
        model_choice = st.selectbox("Select Model", ["lstm", "svm", "logistic_regression"])
        
        if st.button("Start Batch Analysis"):
            with st.spinner("Analyzing..."):
                texts = df['text'].tolist()
                payload = [{"text": t, "model": model_choice} for t in texts]
                
                try:
                    response = requests.post(f"{API_URL}/predict/batch", json=payload)
                    results = response.json()
                    
                    results_df = pd.DataFrame(results)
                    final_df = pd.concat([df, results_df], axis=1)
                    
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_sentiment_distribution(final_df))
                    with col2:
                        st.plotly_chart(plot_confidence_histogram(final_df))
                    
                    st.write("### Detailed Results")
                    st.dataframe(final_df)
                    
                    st.download_button(
                        "Download Results as CSV",
                        final_df.to_csv(index=False),
                        "sentiment_results.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"Error during batch analysis: {e}")
