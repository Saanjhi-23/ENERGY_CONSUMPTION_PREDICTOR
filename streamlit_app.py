import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model
model = joblib.load("random_forest_extended_model.pkl")

# Sidebar inputs
st.sidebar.header("Input Parameters")

temperature = st.sidebar.slider("Temperature", -10, 50, 25)
humidity = st.sidebar.slider("Humidity", 0, 100, 50)
square_footage = st.sidebar.number_input("Square Footage", min_value=100, value=1000)
occupancy = st.sidebar.slider("Occupancy", 0, 100, 10)
hvac_usage = st.sidebar.slider("HVAC Usage", 0, 100, 50)
lighting_usage = st.sidebar.slider("Lighting Usage", 0, 100, 30)
hour = st.sidebar.slider("Hour of Day", 0, 23, datetime.now().hour)
holiday = st.sidebar.selectbox("Is it a holiday?", ["No", "Yes"])
holiday_val = 1 if holiday == "Yes" else 0
day_of_week = datetime.today().weekday()

# Derived features using your original formulas
effective_usage = occupancy * lighting_usage
discomfort_index = temperature + humidity

# Construct input
input_dict = {
    "Temperature": temperature,
    "HVACUsage": hvac_usage,
    "SquareFootage": square_footage,
    "EffectiveUsage": occupancy * lighting_usage,
    "DiscomfortIndex": temperature + humidity
}


input_df = pd.DataFrame([input_dict])

# Predict
prediction = model.predict(input_df)[0]

# Display result
st.title("Energy Consumption Prediction")
st.write(f"### Predicted Energy Usage: `{prediction:.2f}` units")
