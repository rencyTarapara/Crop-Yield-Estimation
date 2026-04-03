import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown

# Page config
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model')

model_path = os.path.join(model_dir, 'model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
encoders_path = os.path.join(model_dir, 'encoders.pkl')
metrics_path = os.path.join(model_dir, 'metrics.pkl')

@st.cache_resource
def load_models():
    os.makedirs(model_dir, exist_ok=True)

    # Google Drive file IDs
    FILES = {
        "model": ("1o63pfzG-3S8OBkcKerJdpDwsqo8eT8rP", model_path),
        # 👉 ADD YOUR FILE IDs BELOW
        "scaler": ("YOUR_SCALER_FILE_ID", scaler_path),
        "encoders": ("YOUR_ENCODERS_FILE_ID", encoders_path),
        "metrics": ("YOUR_METRICS_FILE_ID", metrics_path),
    }

    # Download files if missing
    for name, (file_id, path) in FILES.items():
        if not os.path.exists(path) and "YOUR_" not in file_id:
            try:
                st.info(f"⬇️ Downloading {name}...")
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, path, quiet=False)
            except Exception as e:
                st.error(f"❌ Failed to download {name}: {e}")

    # Load models
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            return None, None, None, None

        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None
        metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else None

        return model, scaler, encoders, metrics

    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None, None


model, scaler, encoders, metrics = load_models()

# UI
st.title("🌾 Agricultural Crop Yield Prediction")
st.markdown("""
This application predicts **crop yield (hg/ha)** based on environmental and agricultural factors.
Configure the parameters below and click predict.
""")

# If model missing
if model is None:
    st.error("⚠️ Model not loaded! Please check Google Drive links.")
    st.stop()

# Sidebar
st.sidebar.header("📊 Model Performance")

if metrics:
    st.sidebar.metric("R² Score", f"{metrics.get('r2_score',0)*100:.2f}%")
    st.sidebar.metric("RMSE", f"{metrics.get('rmse',0):,.0f}")
else:
    st.sidebar.warning("No metrics available")

# Encoders check
if encoders:
    area_encoder = encoders.get('Area')
    item_encoder = encoders.get('Item')

    area_options = area_encoder.classes_ if area_encoder else []
    item_options = item_encoder.classes_ if item_encoder else []
else:
    st.error("⚠️ Encoders missing! Upload encoders.pkl")
    st.stop()

# Inputs
st.header("⚙️ Input Features")

col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("🌍 Area", area_options)
    year = st.number_input("📅 Year", 1990, 2050, 2013)
    rain = st.number_input("🌧️ Rainfall", 0.0, 4000.0, 1000.0)

with col2:
    item = st.selectbox("🌱 Crop", item_options)
    temp = st.number_input("🌡️ Temperature", -10.0, 50.0, 20.0)
    pesticides = st.number_input("🧪 Pesticides", 0.0, 500000.0, 100.0)

# Predict
if st.button("🚀 Predict Yield"):
    try:
        area_encoded = area_encoder.transform([area])[0]
        item_encoded = item_encoder.transform([item])[0]

        input_data = pd.DataFrame({
            'Area': [area_encoded],
            'Item': [item_encoded],
            'Year': [year],
            'average_rain_fall_mm_per_year': [rain],
            'pesticides_tonnes': [pesticides],
            'avg_temp': [temp]
        })

        if scaler:
            input_data[['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp']] = \
                scaler.transform(input_data[['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp']])

        prediction = model.predict(input_data)[0]

        st.success(f"🌾 Predicted Yield: {prediction:,.2f} hg/ha")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
