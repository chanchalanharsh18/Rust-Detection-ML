"""
Streamlit Application - Rust Detection Using Electric Field Analysis
MVP Prototype for predicting rust percentage in materials using electric field measurements.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Rust Detection Using Electric Field Analysis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="auto"
)

# Application paths
MODEL_PATH = os.path.join("models", "rust_model.joblib")
METRICS_PATH = os.path.join("models", "metrics.json")


@st.cache_resource
def load_saved_model(path):
    """Loads the pre-trained Joblib model from disk."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_metrics(path):
    """Loads model evaluation metrics metadata."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# Load artifacts
model = load_saved_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

# Main Application Title
st.title("Rust Detection Using Electric Field Analysis")
st.caption("Machine Learning MVP for Rust Percentage Prediction from Electric Field Measurements")

st.divider()

if model is None:
    st.error(
        "⚠️ Trained model not found at `models/rust_model.joblib`.\n\n"
        "Please run `python train.py` first to train and save the model."
    )
    st.stop()

# Layout: Two tabs or sections (Prediction and Performance)
st.subheader("Field Measurement Inputs")

# Pre-defined test sample presets to help testing
sample_presets = {
    "Custom Input": (0.000000, 0.000000),
    "Sample A (Low E-field)": (-0.009947, 0.009120),
    "Sample B (Mid E-field)": (0.150000, -0.250000),
    "Sample C (High E-field)": (-0.450000, 0.350000),
}

selected_preset = st.selectbox(
    "Choose a preset or enter custom values:",
    options=list(sample_presets.keys()),
    index=0
)

default_re, default_im = sample_presets[selected_preset]

col1, col2 = st.columns(2)

with col1:
    real_field = st.number_input(
        "Real Electric Field (Re):",
        value=float(default_re),
        step=0.01,
        format="%.6f",
        help="Real component of the measured electric field."
    )

with col2:
    imag_field = st.number_input(
        "Imaginary Electric Field (Im):",
        value=float(default_im),
        step=0.01,
        format="%.6f",
        help="Imaginary component of the measured electric field."
    )

st.write("")
predict_clicked = st.button("Predict Rust Percentage", type="primary", use_container_width=True)

if predict_clicked:
    # Prepare input DataFrame matching training feature names
    input_data = pd.DataFrame([{
        "real_electric_field": real_field,
        "imaginary_electric_field": imag_field
    }])
    
    # Run prediction
    raw_prediction = float(model.predict(input_data)[0])
    # Clip between 0% and 100%
    prediction = max(0.0, min(100.0, raw_prediction))
    
    st.markdown("### Prediction Result")
    
    # Result Card
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(
            label="Predicted Rust Percentage",
            value=f"{prediction:.2f}%"
        )
    with res_col2:
        # Progress bar representation
        st.progress(prediction / 100.0)
        
        # Qualitative severity tag
        if prediction < 20.0:
            severity = "Low Rust Severity"
            color = "🟢"
        elif prediction < 50.0:
            severity = "Moderate Rust Severity"
            color = "🟡"
        elif prediction < 75.0:
            severity = "High Rust Severity"
            color = "🟠"
        else:
            severity = "Severe Rust / Critical"
            color = "🔴"
            
        st.write(f"**Classification:** {color} {severity}")

st.divider()

# Model Evaluation Metrics Section
st.subheader("Model Evaluation Metrics")

if metrics:
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric(
            label="MAE (Mean Absolute Error)",
            value=f"{metrics.get('mae', 'N/A')}%",
            help="Average absolute difference between actual and predicted rust percentage on test set."
        )
    with m_col2:
        st.metric(
            label="RMSE (Root Mean Squared Error)",
            value=f"{metrics.get('rmse', 'N/A')}%",
            help="Standard deviation of the prediction residuals on test set."
        )
    with m_col3:
        st.metric(
            label="R² Score (Coefficient of Determination)",
            value=f"{metrics.get('r2_score', 'N/A')}",
            help="Proportion of variance explained by the model."
        )
        
    with st.expander("ℹ️ Dataset & Model Details"):
        st.write(f"- **Algorithm:** {metrics.get('model_type', 'Random Forest Regressor')}")
        st.write(f"- **Train/Test Split:** 80:20 (Grouped by sample identity to prevent data leakage)")
        st.write(f"- **Training Records:** {metrics.get('train_records', 'N/A'):,}")
        st.write(f"- **Testing Records:** {metrics.get('test_records', 'N/A'):,}")
        st.write(f"- **Total Combined Records:** {metrics.get('total_records', 'N/A'):,}")
        st.write(f"- **Sample Materials Count:** {metrics.get('n_samples', 'N/A')}")
        st.write(f"- **Input Features:** `real_electric_field`, `imaginary_electric_field`")
        st.write(f"- **Target:** `rust_percentage` (%)")
else:
    st.info("Metrics file not found. Re-run `train.py` to regenerate metrics metadata.")
