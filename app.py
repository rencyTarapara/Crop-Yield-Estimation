import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown

# Page config
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")

# =========================
# PATHS (FIXED FOR GITHUB STRUCTURE)
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
encoders_path = os.path.join(base_dir, 'encoders.pkl')
metrics_path = os.path.join(base_dir, 'metrics.pkl')

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    # Download model if missing
    if not os.path.exists(model_path):
        try:
            st.info("⬇️ Downloading model from Google Drive...")
            file_id = "1o63pfzG-3S8OBkcKerJdpDwsqo8eT8rP"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None, None, None, None

    try:
        model = joblib.load(model_path)

        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None
        metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else None

        return model, scaler, encoders, metrics

    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        return None, None, None, None


model, scaler, encoders, metrics = load_models()

# =========================
# UI HEADER
# =========================
st.title("🌾 Agricultural Crop Yield Prediction")

st.markdown("""
This application predicts **crop yield (hg/ha)** based on environmental and agricultural factors.  
Fill in the inputs and click **Predict Yield**.
""")

# =========================
# ERROR HANDLING
# =========================
if model is None:
    st.error("⚠️ Model not loaded! Please check your Google Drive link.")
    st.stop()

if encoders is None:
    st.error("⚠️ encoders.pkl not found! Please upload it to GitHub.")
    st.stop()

# =========================
# SIDEBAR (METRICS)
# =========================
st.sidebar.header("📊 Model Performance")

if metrics:
    r2 = metrics.get("r2_score", 0) * 100
    rmse = metrics.get("rmse", 0)

    st.sidebar.metric("R² Score", f"{r2:.2f}%")
    st.sidebar.metric("RMSE", f"{rmse:,.0f}")
else:
    st.sidebar.warning("Metrics not available")

# =========================
# INPUT OPTIONS
# =========================
area_encoder = encoders.get('Area')
item_encoder = encoders.get('Item')

if area_encoder is None or item_encoder is None:
    st.error("⚠️ Encoders missing required features. Retrain model.")
    st.stop()

area_options = area_encoder.classes_
item_options = item_encoder.classes_

# =========================
# INPUT UI
# =========================
st.header("⚙️ Input Features")

col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("🌍 Area (Country/Region)", area_options)
    year = st.number_input("📅 Year", 1990, 2050, 2013)
    rain = st.number_input("🌧️ Average Rainfall (mm/year)", 0.0, 4000.0, 1000.0)

with col2:
    item = st.selectbox("🌱 Crop", item_options)
    temp = st.number_input("🌡️ Average Temperature (°C)", -10.0, 50.0, 20.0)
    pesticides = st.number_input("🧪 Pesticides (tonnes)", 0.0, 500000.0, 100.0)

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Yield", use_container_width=True):
    try:
        # Encode categorical
        area_encoded = area_encoder.transform([area])[0]
        item_encoded = item_encoder.transform([item])[0]

        # Create dataframe
        input_data = pd.DataFrame({
            'Area': [area_encoded],
            'Item': [item_encoded],
            'Year': [year],
            'average_rain_fall_mm_per_year': [rain],
            'pesticides_tonnes': [pesticides],
            'avg_temp': [temp]
        })

        # Scale if scaler exists
        if scaler:
            input_data[['Year',
                        'average_rain_fall_mm_per_year',
                        'pesticides_tonnes',
                        'avg_temp']] = scaler.transform(
                input_data[['Year',
                            'average_rain_fall_mm_per_year',
                            'pesticides_tonnes',
                            'avg_temp']]
            )

        # Predict
        with st.spinner("🔍 Analyzing data..."):
            prediction = model.predict(input_data)[0]

        # Output
        st.success("### 📊 Predicted Yield")
        st.write(f"## {prediction:,.2f} hg/ha")

        st.info("💡 Yield is measured in **hectograms per hectare (hg/ha)**.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
