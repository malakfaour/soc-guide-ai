<<<<<<< HEAD
import streamlit as st
import pandas as pd
from src.models.lightgbm.predict import LightGBMPredictor

# Page config
st.set_page_config(
    page_title="AI Security Alert Classifier",
    layout="wide"
)

# Title
st.title("🔐 AI Security Incident Classifier")
st.write("Predict whether an alert is TruePositive, FalsePositive, or Benign")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Upload your dataset and classify alerts instantly")

# Load model (cached)
@st.cache_resource
def load_model():
    return LightGBMPredictor()

predictor = load_model()

# Upload section
st.header("📂 Upload Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    if st.button("🚀 Predict"):
        results = predictor.predict(df)

        predictions = results['predictions']
        probs = results['probabilities']

        st.success("✅ Predictions completed!")

        # Mapping classes
        class_map = {
            0: "FalsePositive",
            1: "BenignPositive",
            2: "TruePositive"
        }

        # Create output dataframe
        output_df = df.copy()

        output_df["Prediction"] = [class_map[p] for p in predictions]
        output_df["Confidence"] = probs.max(axis=1)

        # Add probability columns
        prob_df = pd.DataFrame(
            probs,
            columns=["FalsePositive", "BenignPositive", "TruePositive"]
        )

        output_df = pd.concat([output_df, prob_df], axis=1)

        # Show results
        st.subheader("📊 Results Preview")
        st.dataframe(output_df.head())

        # Show chart
        st.subheader("📊 Prediction Probabilities (First 10 rows)")
        st.bar_chart(prob_df.head(10))

        # Show single sample nicely
        st.subheader("🔍 Example Prediction")
        sample = output_df.iloc[0]

        st.write(f"**Prediction:** {sample['Prediction']}")
        st.write(f"**Confidence:** {sample['Confidence']:.2f}")

        # Download button
        st.download_button(
            label="📥 Download Results",
            data=output_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

else:
    st.info("📥 Upload a CSV file to start prediction")
=======
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from xgboost import XGBClassifier

app = FastAPI()

# Load model once
model = XGBClassifier()
model.load_model("models/xgboost_model.json")


class InputData(BaseModel):
    features: list  # 44 features


@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)

    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))

    return {
        "prediction": pred,
        "probabilities": proba.tolist()
    }
>>>>>>> origin/main
