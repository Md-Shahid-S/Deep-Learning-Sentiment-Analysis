import streamlit as st

def set_page_style():
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7f9;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

def show_header(title, subtitle):
    st.title(title)
    st.markdown(f"**{subtitle}**")
    st.divider()
