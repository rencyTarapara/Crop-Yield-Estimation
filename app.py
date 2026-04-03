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
model_path = os.path.join(base_dir, 'model', 'model.pkl')
scaler_path = os.path.join(base_dir, 'model', 'scaler.pkl')
encoders_path = os.path.join(base_dir, 'model', 'encoders.pkl')
metrics_path = os.path.join(base_dir, 'model', 'metrics.pkl')

@st.cache_resource
def load_models():
    # If model doesn't exist, download it dynamically from Google Drive
    if not os.path.exists(model_path):
        import gdown
        st.info("Downloading large model file from Google Drive... Please wait.")
        # NOTE: The User must replace 'YOUR_GOOGLE_DRIVE_FILE_ID' below 
        # with the exact file ID from their Google Drive shareable link!
        file_id = '1o63pfzG-3S8OBkcKerJdpDwsqo8eT8rP'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)
        
        metrics = None
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            
        return model, scaler, encoders, metrics
    else:
        return None, None, None, None

model, scaler, encoders, metrics = load_models()

# UI elements
st.title("🌾 Agricultural Crop Yield Prediction")
st.markdown("""
This application predicts **crop yield (hg/ha)** based on environmental and agricultural factors.
Configure the parameters below and click predict to get a yield estimate.
""")

if model is None:
    st.error("⚠️ Model files not found! Please ensure training has completed (`python model/train.py`).")
else:
    # Sidebar for Model Accuracy
    st.sidebar.header("📊 Model Performance")
    if metrics:
        accuracy = metrics.get('r2_score', 0) * 100
        rmse = metrics.get('rmse', 0)
        
        st.sidebar.metric("Model Accuracy (R²)", f"{accuracy:.2f}%")
        st.sidebar.metric("Evaluation Error (RMSE)", f"{rmse:,.0f}")
        
        st.sidebar.markdown("""
        ---
        **Note**: The accuracy represents how well the model explains the variance in crop yields.
        """)
    else:
        st.sidebar.warning("Metrics not found. Please retrain the model.")
    # Ensure encoders have the valid labels
    area_encoder = encoders.get('Area')
    item_encoder = encoders.get('Item')
    
    if area_encoder and item_encoder:
        area_options = area_encoder.classes_
        item_options = item_encoder.classes_
    else:
        area_options = []
        item_options = []
        st.error("Encoders missing expected features. Please retrain the model.")
        
    st.header("⚙️ Input Features")
    
    # Form layout
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            area = st.selectbox("🌍 Area (Country/Region)", area_options)
            year = st.number_input("📅 Year", min_value=1990, max_value=2050, value=2013, step=1)
            rain = st.number_input("🌧️ Average Rainfall (mm/year)", min_value=0.0, max_value=4000.0, value=1000.0, step=10.0)
            
        with col2:
            item = st.selectbox("🌱 Item (Crop)", item_options)
            temp = st.number_input("🌡️ Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
            pesticides = st.number_input("🧪 Pesticides (tonnes)", min_value=0.0, max_value=500000.0, value=100.0, step=10.0)
            
    st.markdown("---")
    
    # Predict button logic
    col_empty1, col_center, col_empty2 = st.columns([1, 2, 1])
    with col_center:
        predict_btn = st.button("🚀 Predict Yield", use_container_width=True)
    
    if predict_btn:
        try:
            # Prepare input data
            area_encoded = area_encoder.transform([area])[0]
            item_encoded = item_encoder.transform([item])[0]
            
            continuous_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
            
            input_data = pd.DataFrame({
                'Area': [area_encoded],
                'Item': [item_encoded],
                'Year': [year],
                'average_rain_fall_mm_per_year': [rain],
                'pesticides_tonnes': [pesticides],
                'avg_temp': [temp]
            })
            
            # Scale continuous features
            input_data[continuous_features] = scaler.transform(input_data[continuous_features])
            
            # Predict
            with st.spinner("Analyzing..."):
                prediction = model.predict(input_data)[0]
            
            # Display Result
            st.success("### 📊 Predicted Yield")
            st.write(f"## {prediction:,.2f} hg/ha")
            st.info("💡 **Formula Note**: The prediction is measured in Hectograms per Hectare (hg/ha).")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
