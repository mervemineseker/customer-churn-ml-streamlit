import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from pathlib import Path
import urllib.request


def prettify_feature_name(name: str) -> str:
    # Remove sklearn prefixes like "num__" or "cat__"
    if "__" in name:
        _, core = name.split("__", 1)
    else:
        core = name

    # OneHot style often looks like "Contract_Month-to-month"
    # Convert to "Contract = Month-to-month"
    if "_" in core:
        parts = core.split("_")
        feature = parts[0]
        value = "_".join(parts[1:])  # keep possible underscores in value
        return f"{feature} = {value}"

    return core

MODEL_PATH = "models/churn_pipeline.joblib"
RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TELCO_CSV_URL = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction (Demo)")
st.caption("Predict churn probability and show approximate explanations using SHAP.")


import os
import subprocess
import joblib
import streamlit as st

MODEL_PATH = "models/churn_pipeline.joblib"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found. Training model...")
        subprocess.run(["python", "src/train.py"], check=True)


    return joblib.load(MODEL_PATH)






def prettify_feature_name(name: str) -> str:
    """
    Convert sklearn feature names to human-friendly labels.
    Examples:
      - "cat__Contract_Month-to-month" -> "Contract = Month-to-month"
      - "num__MonthlyCharges" -> "MonthlyCharges"
    """
    # Remove sklearn prefixes like "num__" or "cat__"
    if "__" in name:
        _, core = name.split("__", 1)
    else:
        core = name

    # Convert OneHot-style "Feature_Value" -> "Feature = Value"
    if "_" in core:
        parts = core.split("_")
        feature = parts[0]
        value = "_".join(parts[1:])
        return f"{feature} = {value}"

    return core


def icon_for_feature(label: str) -> str:
    s = label.lower()
    if "contract" in s:
        return "📄"
    if "tenure" in s:
        return "⏳"
    if "charges" in s:
        return "💳"
    if "internet" in s or "streaming" in s:
        return "🌐"
    if "techsupport" in s or "support" in s:
        return "🛠️"
    if "security" in s:
        return "🔒"
    if "backup" in s:
        return "🗄️"
    return "🔎"


pipe = load_pipeline()

st.subheader("Customer Inputs")

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

X = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}])


if st.button("Predict churn probability"):
        # Predict
        proba = float(pipe.predict_proba(X)[:, 1][0])
        st.metric("Churn probability", f"{proba * 100:.1f}%")
        st.write("Risk level:", "HIGH" if proba >= 0.5 else "LOW")

        st.divider()
        st.subheader("Top reasons (approx.)")
        st.write("Positive impact increases churn risk; negative impact decreases it.")

        # Extract pipeline components
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]

        # Transform input
        X_trans = pre.transform(X)




       # SHAP explanation (cloud-safe, pipeline-level)
bg_sample = pd.concat([X] * 50, ignore_index=True)
explainer = shap.Explainer(pipe, bg_sample)
shap_values = explainer(X)

sv = shap_values.values[0]
feature_names = shap_values.feature_names

# Top drivers
top_k = 8
top_idx = np.argsort(np.abs(sv))[::-1][:top_k]

top = pd.DataFrame({
    "feature": np.array(feature_names)[top_idx],
    "impact": sv[top_idx],
})


                # Make features human-friendly
top["feature"] = top["feature"].astype(str).apply(prettify_feature_name)

        # Cards for top 5
top5 = top.head(5).copy()
cols = st.columns(5)

        for i, (_, row) in enumerate(top5.iterrows()):
            label = str(row["feature"])
            impact = float(row["impact"])

            direction = "Increases risk" if impact > 0 else "Decreases risk"
            emoji = icon_for_feature(label)

            with cols[i]:
                st.markdown(
                    f"""
<div style="border-radius:16px; padding:14px; border:1px solid rgba(0,0,0,0.12);">
  <div style="font-size:22px; line-height:1;">{emoji}</div>
  <div style="margin-top:6px; font-weight:600;">{label}</div>
  <div style="margin-top:6px; font-size:13px; opacity:0.85;">{direction}</div>
  <div style="margin-top:8px; font-size:12px; opacity:0.75;">Impact: {impact:.4f}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("All top features")
        st.dataframe(top, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed. Please check inputs and model compatibility.")
        st.exception(e)
