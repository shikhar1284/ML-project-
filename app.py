import pandas as pd
import streamlit as st


st.title("Naive Bayes Visualization App")
st.write("Explore the performance of Naive Bayes with interactive visualizations.")

st.sidebar.header("Settings")
var_smoothing = st.sidebar.slider(
    "Variance Smoothing",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
)
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Breast Cancer Wisconsin", "Wine"],
)
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "txt"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_table(uploaded_file)
        st.warning("Dataset loaded successfully (as a table).")
