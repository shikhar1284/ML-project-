import pandas as pd
import streamlit as st
from sklearn.naive_bayes import GaussianNB


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

if df is not None:
    try:
        if df.shape[1] < 2:
            st.warning("The dataset must include at least one feature column and one target column.")
        else:
            target_name = df.columns[-1]
            X = df.drop(columns=[target_name])
            y = df[target_name]

            X = pd.get_dummies(X, drop_first=False)
            X = X.apply(pd.to_numeric, errors="coerce")

            model_data = pd.concat([X, y.rename(target_name)], axis=1).dropna()
            X = model_data.drop(columns=[target_name])
            y = model_data[target_name]

            if X.empty or y.empty:
                st.warning("Error loading or processing the dataset.")
            else:
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X, y)

                st.success("Model trained successfully!")
                st.write(f"Target column: {target_name}")
    except Exception:
        st.warning("Error loading or processing the dataset.")
