import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")

st.title("🌾 Crop Yield Prediction Web App")
st.write("Predict expected crop yield (tons/hectare) using climate and soil conditions.")

# Input fields
year = st.number_input("Year", 2024, 2050)
crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Soybean"])
avg_temp = st.number_input("Average Temperature (°C)", 10.0, 50.0)
avg_humidity = st.number_input("Average Humidity (%)", 0, 100)
rainfall = st.number_input("Rainfall (mm)", 0, 3000)
soil_moisture = st.number_input("Soil Moisture (%)", 0, 100)
soil_type = st.selectbox("Soil Type", ["Clay", "Loamy", "Sandy"])
ph = st.number_input("Soil pH Level", 0.0, 14.0)
sunlight = st.number_input("Sunlight (hrs/day)", 0.0, 14.0)

# Encodings
crop_map = {"Wheat": 3, "Rice": 2, "Maize": 1, "Soybean": 0}
soil_map = {"Clay": 0, "Loamy": 1, "Sandy": 2}

if st.button("Predict Yield"):
    input_data = np.array([[year,
                            crop_map[crop],
                            avg_temp,
                            avg_humidity,
                            rainfall,
                            soil_moisture,
                            soil_map[soil_type],
                            ph,
                            sunlight]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.success(f"🌱 Predicted Yield: {prediction:.2f} tons/hectare")