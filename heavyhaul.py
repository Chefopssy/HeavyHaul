import streamlit as st
import numpy as np
import os
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
from torch_model_def import DisruptionPredictor  # Task 4 Class

st.set_page_config(page_title="HeavyHaul AI", layout="centered")
st.title("🚚 HeavyHaul AI — Logistics Intelligence Platform")

# Load all models with caching
@st.cache_resource
def load_models():
    models = {}
    try:
        models["task3"] = load_model("models/task3_delivery_nn.keras")
        st.success("✅ Task 3 model loaded")
    except Exception as e:
        st.error(f"❌ Task 3: {e}")

    try:
        models["task4"] = DisruptionPredictor()
        models["task4"].load_state_dict(torch.load("models/task4_disruption_model.pt", map_location=torch.device('cpu')))
        models["task4"].eval()
        st.success("✅ Task 4 model loaded")
    except Exception as e:
        st.error(f"❌ Task 4: {e}")

    try:
        models["task5"] = load_model("models/task5_timeseries_lstm.keras")
        st.success("✅ Task 5 model loaded")
    except Exception as e:
        st.error(f"❌ Task 5: {e}")

    try:
        models["task6"] = load_model("models/task6_package_cnn.keras")
        st.success("✅ Task 6 model loaded")
    except Exception as e:
        st.error(f"❌ Task 6: {e}")

    try:
        models["task7"] = load_model("models/task7_anomaly_autoencoder.keras")
        st.success("✅ Task 7 model loaded")
    except Exception as e:
        st.error(f"❌ Task 7: {e}")

    try:
        models["task8"] = load_model("models/task8_maintenance_classifier.keras")
        st.success("✅ Task 8 model loaded")
    except Exception as e:
        st.error(f"❌ Task 8: {e}")

    return models

models = load_models()

# Sidebar task selector
selected_task = st.sidebar.selectbox("🔍 Select Task", [
    "📦 Task 3: Delivery Time",
    "🚨 Task 4: Disruption",
    "📈 Task 5: Forecast",
    "🧪 Task 6: Package Inspection",
    "⚠️ Task 7: Anomaly Detection",
    "🛠️ Task 8: Maintenance Prediction"
])

# Task 3 — Delivery Time Prediction
if selected_task == "📦 Task 3: Delivery Time":
    st.subheader("📦 Predict Delivery Time (in days)")
    weight = st.number_input("Weight")
    volume = st.number_input("Volume")
    distance = st.number_input("Distance (km)")
    traffic = st.selectbox("Traffic Level", [0, 1, 2])  # Low, Medium, High
    priority = st.selectbox("Shipping Priority", [0, 1])  # Standard, Express
    X = np.array([[weight, volume, distance, traffic, priority]])
    pred = models["task3"].predict(X)[0][0]
    st.write(f"📦 Estimated Delivery Time: {pred:.2f} days")

# Task 4 — Disruption Classifier
elif selected_task == "🚨 Task 4: Disruption Prediction":
    st.subheader("🚨 Predict Shipment Disruption")
    route_encoded = st.number_input("Route Code", value=1)
    volume = st.number_input("Volume", value=2.0)
    weather_encoded = st.number_input("Weather Code", value=0)
    X = torch.tensor([[route_encoded, volume, weather_encoded]], dtype=torch.float32)
    with torch.no_grad():
        out = models["task4"](X)
        result = (out > 0.5).float().item()
        st.write("⚠️ Disruption Detected" if result else "✅ Normal Delivery")

# Task 5 — Time Series Forecast
elif selected_task == "📈 Task 5: Forecast":
    st.subheader("📈 Forecast Avg Delivery Time (Next Day)")
    try:
        series = np.loadtxt("task5_timeseries_dataset.csv", delimiter=",", skiprows=1, usecols=1).reshape(-1, 1)
        scaler = StandardScaler()
        series_scaled = scaler.fit_transform(series)
        last_5 = series_scaled[-5:].reshape(1, 5, 1)
        pred = models["task5"].predict(last_5)[0][0]
        result = scaler.inverse_transform([[pred]])[0][0]
        st.write(f"📦 Forecasted Delivery Time: {result:.2f} days")
    except Exception as e:
        st.error(f"Failed to load time series data: {e}")

# --------------------------
# Task 6 — Image Classifier
# --------------------------
elif selected_task == "📦 Task 6: Package Image Inspection":
    st.subheader("🧰 Inspect Package Image (OK / Damaged)")

    uploaded_file = st.file_uploader("Upload Package Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        from PIL import Image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Package", use_column_width=True)

        # Preprocess image
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        try:
            task6_model = load_model("models/task6_package_cnn.keras")
            prediction = task6_model.predict(img_array)
            label = "Damaged" if prediction[0][0] > 0.5 else "OK"
            st.success(f"🧠 Model Prediction: **{label}**")
        except Exception as e:
            st.error(f"❌ Could not load Task 6 model: {e}")

# --------------------------
# Task 7 — Anomaly Detection
# --------------------------
elif selected_task == "🚨 Task 7: Shipment Anomaly Detection":
    st.subheader("🔎 Detect Shipment Anomalies")

    weight = st.number_input("Package Weight", value=50.0)
    volume = st.number_input("Package Volume", value=100.0)
    dim_volume = st.number_input("Dimensional Volume (L x W x H)", value=150.0)

    features = np.array([[weight, volume, dim_volume]])

    try:
        from sklearn.preprocessing import StandardScaler
        task7_model = load_model("models/task7_anomaly_autoencoder.keras")
        scaler = joblib_load("models/task7_scaler.joblib")  # Save this during training if needed

        features_scaled = scaler.transform(features)
        reconstruction = task7_model.predict(features_scaled)
        error = np.mean(np.square(features_scaled - reconstruction), axis=1)

        threshold = 0.1  # Use same threshold used during training
        is_anomaly = error > threshold

        st.write(f"🔁 Reconstruction Error: {error[0]:.4f}")
        st.success("✅ No Anomaly Detected") if not is_anomaly else st.error("🚨 Anomaly Detected!")
    except Exception as e:
        st.error(f"❌ Task 7 model/scaler load error: {e}")

# --------------------------
# Task 8 — Predictive Maintenance
# --------------------------
elif selected_task == "🛠️ Task 8: Predictive Maintenance":
    st.subheader("🔧 Predict Maintenance Risk")

    engine_hours = st.number_input("Engine Hours", value=1200.0)
    mileage = st.number_input("Mileage (km)", value=50000.0)
    fuel = st.number_input("Fuel Consumption (L)", value=1500.0)
    temperature = st.number_input("Temperature (°C)", value=85.0)
    vibration = st.number_input("Vibration Level", value=15.0)

    input_data = np.array([[engine_hours, mileage, fuel, temperature, vibration]])

    try:
        task8_model = load_model("models/task8_maintenance_classifier.keras")
        scaler = joblib_load("models/task8_scaler.joblib")  # Save this during training if needed

        input_scaled = scaler.transform(input_data)
        pred = task8_model.predict(input_scaled)
        result = "⚠️ Breakdown Likely" if pred[0][0] > 0.5 else "✅ No Issue Detected"
        st.success(result)
    except Exception as e:
        st.error(f"❌ Could not load Task 8 model or scaler: {e}")
