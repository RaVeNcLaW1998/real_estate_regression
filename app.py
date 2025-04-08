import streamlit as st
import pandas as pd
import numpy as np
import joblib
from real_estate.modeling.train import (
    load_data,
    split_features_target,
    train_linear_model,
)

st.set_page_config(page_title="🏠 Real Estate Price Predictor", layout="centered")

st.title("🏠 Real Estate Price Predictor")


# Load data
@st.cache_data
def load_and_train_model():
    df = load_data("data/raw/final.csv")
    X, y = split_features_target(df)
    model = train_linear_model(X, y)
    return model, X.columns.tolist()


try:
    model, feature_list = load_and_train_model()

    st.sidebar.header("Enter Property Details")

    user_input = {}
    for feature in feature_list:
        user_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    if st.button("Predict Price"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ${prediction:,.2f}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Make sure `final.csv` is placed in `data/raw/`.")
