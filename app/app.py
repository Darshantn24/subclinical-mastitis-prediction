import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Mastitis Prediction",
    page_icon="🐄",
    layout="centered"
)

st.title("🐄 Subclinical Mastitis Prediction")
st.write("AI-based prediction using machine learning.")

# st.info("Application setup successful. Model integration is the next step.")

# --------------------------------------
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_random_forest.pkl"

MODEL_URL = (
    "https://github.com/Darshantn24/"
    "subclinical-mastitis-prediction/releases/download/"
    "model-v1.0.0/baseline_random_forest.pkl"
)

try:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        response = requests.get(MODEL_URL, timeout=60)
        response.raise_for_status()
        MODEL_PATH.write_bytes(response.content)

    model = joblib.load(MODEL_PATH)
    st.success("Trained Random Forest model loaded successfully!")

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()



st.header("Cow Health Parameters")

col1, col2 = st.columns(2)

with col1:
    parity = st.number_input("Parity", 1, 10, 2, key="parity")
    days_in_milk = st.number_input("Days in Milk", 1, 400, 100, key="days_in_milk")
    milk_conductivity = st.number_input("Milk Conductivity", 0.0, 15.0, 5.5, key="milk_conductivity")
    milk_yield = st.number_input("Milk Yield (L/day)", 0.0, 60.0, 25.0, key="milk_yield")
    milk_pH = st.number_input("Milk pH", 4.0, 10.0, 6.6, key="milk_pH")

with col2:
    udder_skin_temp = st.number_input("Udder Skin Temperature", 25.0, 45.0, 36.5, key="udder_skin_temp")
    rectal_temp = st.number_input("Rectal Temperature", 30.0, 45.0, 38.3, key="rectal_temp")
    heart_rate = st.number_input("Heart Rate", 30, 150, 65, key="heart_rate")
    ambient_temp = st.number_input("Ambient Temperature", 0.0, 50.0, 28.0, key="ambient_temp")
    ambient_humidity = st.number_input("Ambient Humidity (%)", 0.0, 100.0, 65.0, key="ambient_humidity")
# -----------------------------------

st.divider()

if st.button(
    "Predict Mastitis",
    type="primary",
    use_container_width=True,
    key="predict_mastitis"
):

    if milk_yield <= 0:
        st.warning("Milk yield must be greater than zero.")
        st.stop()

    input_data = pd.DataFrame([{
        "parity": parity,
        "days_in_milk": days_in_milk,
        "milk_conductivity": milk_conductivity,
        "milk_yield": milk_yield,
        "milk_pH": milk_pH,
        "udder_skin_temp": udder_skin_temp,
        "rectal_temp": rectal_temp,
        "heart_rate": heart_rate,
        "ambient_temp": ambient_temp,
        "ambient_humidity": ambient_humidity
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Results")

    if st.button("Reset Inputs"):
        defaults = {
            "parity": 2,
            "days_in_milk": 100,
            "milk_conductivity": 5.5,
            "milk_yield": 25.0,
            "milk_pH": 6.6,
            "udder_skin_temp": 36.5,
            "rectal_temp": 38.3,
            "heart_rate": 65,
            "ambient_temp": 28.0,
            "ambient_humidity": 65.0
        }

        def reset_inputs():
            for key, value in defaults.items():
                st.session_state[key] = value

        st.rerun()

    if prediction == 1:
        st.error("Mastitis risk pattern detected")
    else:
        st.success("No mastitis risk pattern detected")

    st.divider()

    st.write(f"Model-predicted positive class score: {probability:.2%}")
    st.progress(float(probability))

    if probability >= 0.70:
        st.info("High model score")
    elif probability >= 0.40:
        st.warning("Moderate model score")
    else:
        st.info("Low model score")

    st.caption(
        "This score reflects the model's output on synthetic training data. "
        "It is not a clinically validated probability of disease."
    )

st.caption(
    "Research prototype only. Predictions are based on synthetic "
    "data and are not a substitute for veterinary diagnosis."
)



