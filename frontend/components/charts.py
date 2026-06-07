import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_sentiment_distribution(df):
    counts = df['sentiment'].value_counts().reset_index()
    counts.columns = ['sentiment', 'count']
    fig = px.pie(counts, values='count', names='sentiment', 
                 color='sentiment',
                 color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'},
                 title='Sentiment Distribution')
    return fig

def plot_confidence_histogram(df):
    fig = px.histogram(df, x='confidence', color='sentiment',
                       nbins=20, title='Confidence Distribution',
                       color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c'})
    return fig
