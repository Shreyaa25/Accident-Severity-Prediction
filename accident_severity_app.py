import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_traffic_accident_prediction1.csv")
    df = df[["Weather", "Driver_Alcohol", "Speed_Limit", "Road_Condition", "Accident_Severity"]]
    df.dropna(inplace=True)

    # Encode target
    severity_le = LabelEncoder()
    df["Accident_Severity"] = severity_le.fit_transform(df["Accident_Severity"])

    # Encode features
    feature_encoders = {}
    for col in ["Weather", "Road_Condition"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        feature_encoders[col] = le

    if df["Driver_Alcohol"].dtype == 'object':
        le = LabelEncoder()
        df["Driver_Alcohol"] = le.fit_transform(df["Driver_Alcohol"])
        feature_encoders["Driver_Alcohol"] = le

    return df, feature_encoders, severity_le

df, feature_encoders, severity_le = load_data()

# Train model
X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# UI
st.title(" Accident Severity Prediction")

# User Input
weather_input = st.selectbox("Weather", feature_encoders["Weather"].classes_)
alcohol_input = st.number_input("Driver Alcohol Level (0.0 if none)", min_value=0.0, max_value=1.0, step=0.01)
speed_input = st.number_input("Speed Limit (km/h)", min_value=0, max_value=200, step=5)
road_input = st.selectbox("Road Condition", feature_encoders["Road_Condition"].classes_)

# Prediction
if st.button("Predict Accident Severity"):
    input_data = {
        "Weather": feature_encoders["Weather"].transform([weather_input])[0],
        "Driver_Alcohol": alcohol_input,
        "Speed_Limit": speed_input,
        "Road_Condition": feature_encoders["Road_Condition"].transform([road_input])[0],
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    severity = severity_le.inverse_transform([prediction])[0]

    st.success(f"Predicted Accident Severity: **{severity}**")
