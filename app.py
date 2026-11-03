from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Keep these files in the same folder as app.py.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_model.pkl"
FEATURES_PATH = BASE_DIR / "breast_cancer_features.pkl"

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide",
)

st.title("Breast Cancer Classification")
st.write(
    "This educational app uses a trained machine-learning model to classify "
    "tumor measurements as benign or malignant."
)
st.warning(
    "Educational demonstration only. This tool is not a medical device and "
    "must not be used to diagnose or guide treatment. Consult a qualified "
    "healthcare professional for medical decisions."
)

@st.cache_resource
def load_artifacts():
    """Load the model and the exact feature names used during training."""
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, list(features)

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error(
        "Model files were not found. Put `breast_cancer_model.pkl` and "
        "`breast_cancer_features.pkl` in the same folder as `app.py`."
    )
    st.stop()

try:
    model, feature_names = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the saved model files: {exc}")
    st.stop()

st.subheader("Enter tumor measurements")
st.caption(
    "Enter numeric values in the same units and format used by the training dataset. "
    "The app uses the saved feature list to keep the input columns in the correct order."
)

# Split the inputs into three columns to make the form easier to navigate.
columns = st.columns(3)
input_values = {}

with st.form("prediction_form"):
    for index, feature in enumerate(feature_names):
        with columns[index % 3]:
            input_values[feature] = st.number_input(
                label=str(feature),
                value=0.0,
                format="%.6f",
                key=f"feature_{index}",
                help="Replace 0 with the corresponding measured value.",
            )

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    input_df = pd.DataFrame(
        [[input_values[feature] for feature in feature_names]],
        columns=feature_names,
    )

    try:
        prediction = model.predict(input_df)[0]

        # The commonly used Wisconsin breast-cancer dataset encodes
        # malignant as M and benign as B. Handle either labels or 0/1 output.
        label_map = {
            "M": "Malignant",
            "B": "Benign",
            1: "Malignant",
            0: "Benign",
            "malignant": "Malignant",
            "benign": "Benign",
        }
        result = label_map.get(prediction, label_map.get(str(prediction).lower(), str(prediction)))

        if result == "Malignant":
            st.error(f"Model output: {result}")
        elif result == "Benign":
            st.success(f"Model output: {result}")
        else:
            st.info(f"Model output: {result}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(getattr(model, "classes_", []))
            if classes:
                st.subheader("Model probability estimates")
                for class_label, probability in zip(classes, probabilities):
                    readable_label = label_map.get(
                        class_label,
                        label_map.get(str(class_label).lower(), str(class_label)),
                    )
                    st.write(f"{readable_label}: {probability:.1%}")

        st.caption(
            "These are model outputs, not a clinical diagnosis. Prediction quality "
            "depends on the training data, preprocessing, and input measurements."
        )
    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that the saved model and feature list "
            f"match the training code. Details: {exc}"
        )
