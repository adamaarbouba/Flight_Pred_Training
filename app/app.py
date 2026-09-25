import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "core" / "flights.csv"
MODEL_PATH = BASE_DIR / "models" / "FR_model.joblib"

STOP_MAP = {"zero": 0, "one": 1, "two_or_more": 2}
CLASS_MAP = {"Economy": 0, "Business": 1}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["Unnamed: 0"], errors="ignore")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# Rebuild the same preprocessing used during training.
training_data = df.copy()
training_data["stops"] = training_data["stops"].map(STOP_MAP)
training_data["class"] = training_data["class"].map(CLASS_MAP)
encoded = pd.get_dummies(
    training_data,
    columns=["departure_time", "arrival_time", "airline"],
    dtype=int,
)
scaler = MinMaxScaler()
scaler.fit(encoded[["duration", "days_left"]])

feature_columns = (
    encoded.drop(
        columns=["source_city", "flight", "destination_city", "price"],
        errors="ignore",
    ).columns.tolist()
)

st.set_page_config(page_title="Flight Price Predictor", page_icon="")
st.title("Flight Price Predictor")

with st.form("prediction_form"):
    airline = st.selectbox("Airline", sorted(df["airline"].dropna().unique()))
    departure = st.selectbox("Departure time", sorted(df["departure_time"].dropna().unique()))
    arrival = st.selectbox("Arrival time", sorted(df["arrival_time"].dropna().unique()))
    stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
    flight_class = st.selectbox("Class", ["Economy", "Business"])
    duration = st.number_input("Duration (hours)", min_value=0.1, value=2.5, step=0.1)
    days_left = st.number_input("Days left", min_value=1, value=30, step=1)
    submitted = st.form_submit_button("Predict price")

if submitted:
    row = {column: 0 for column in feature_columns}
    row["stops"] = STOP_MAP[stops]
    row["class"] = CLASS_MAP[flight_class]
    row["duration"], row["days_left"] = scaler.transform([[duration, days_left]])[0]

    for column in (
        f"airline_{airline}",
        f"departure_time_{departure}",
        f"arrival_time_{arrival}",
    ):
        if column in row:
            row[column] = 1

    features = pd.DataFrame([row], columns=feature_columns)
    price = model.predict(features)[0]
    st.success(f"Estimated price: ₹{price:,.0f}")
