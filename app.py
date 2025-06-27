import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("discrepancy_predictor_model.pkl")

# Label encoder mapping (sesuai saat training)
item_type_mapping = {'spare_part': 2, 'consumable': 0, 'machine': 1}
storage_location_mapping = {'zone_A': 2, 'zone_B': 0, 'zone_C': 1}

st.title("📦 Discrepancy Prediction Dashboard")
st.markdown("Gunakan form ini untuk memprediksi apakah aktivitas gudang berisiko menyebabkan discrepancy.")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    item_type = st.selectbox("Item Type", list(item_type_mapping.keys()))
    volume_per_day = st.slider("Volume per Day", 1, 100, 50)
    storage_location = st.selectbox("Storage Location", list(storage_location_mapping.keys()))
    picker_id = st.slider("Picker ID", 1000, 1100, 1050)
    picking_time_min = st.slider("Picking Time (minutes)", 1.0, 30.0, 15.0)
    binning_time_min = st.slider("Binning Time (minutes)", 1.0, 20.0, 10.0)
    error_history = st.slider("Error History Count", 0, 10, 1)

# Prepare input for model
input_data = pd.DataFrame([{
    'item_type': item_type_mapping[item_type],
    'volume_per_day': volume_per_day,
    'storage_location': storage_location_mapping[storage_location],
    'picker_id': picker_id,
    'picking_time_min': picking_time_min,
    'binning_time_min': binning_time_min,
    'error_history': error_history
}])

# Prediction
if st.button("Predict Discrepancy"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Prediksi: DISCREPANCY mungkin terjadi! (Probabilitas: {probability:.2f})")
    else:
        st.success(f"✅ Prediksi: Aman, tidak ada discrepancy. (Probabilitas: {probability:.2f})")

    st.markdown("---")
    st.subheader("Detail Input")
    st.write(input_data)