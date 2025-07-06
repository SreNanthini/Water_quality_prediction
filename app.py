# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

# Load the model and structure
model = joblib.load("pollution_model_compressed.pkl")
model_cols = joblib.load("model_columns.pkl")

# Let's create an User interface
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

# User inputs
year_input = st.text_input("Enter Year (between 2000 and 2100)")
station_input = st.text_input("Enter Station ID (1 to 22)")


# VALID Station IDs used during training
valid_station_ids = list(range(1, 23))  # IDs 1 to 22

if st.button('Predict'):
    if not year_input.isdigit() or not station_input.isdigit():
        st.error("❌ Please enter numbers for both Year and Station ID.")
        st.subheader("⚠️ Prediction not possible. Correct the inputs.")
    else:
        year = int(year_input)
        station_id = int(station_input)

        if not (2000 <= year <= 2100):
            st.error("❌ Year must be between 2000 and 2100.")
            st.subheader("⚠️ Prediction not possible. Correct the inputs.")
        elif not (1 <= station_id <= 22):
            st.error("❌ Station ID must be between 1 and 22.")
            st.subheader("⚠️ Prediction not possible. Correct the inputs.")
        else:
            # Valid input → Predict
            input_df = pd.DataFrame({'year': [year], 'id': [station_id]})
            input_encoded = pd.get_dummies(input_df, columns=['id'])

            for col in model_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_cols]

            predicted_pollutants = model.predict(input_encoded)[0]
            pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

            st.subheader(f"✅ Predicted pollutant levels for Station ID {station_id} in {year}:")
            for p, val in zip(pollutants, predicted_pollutants):
                st.write(f"{p}: {val:.2f}")




# visualizations 
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("---")
st.header("📊 Visualize Water Quality Data")

# load data for plotting
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/SreNanthini/Water_quality_prediction/main/afa2e701598d20110228.csv"
    df = pd.read_csv(url, sep=';')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

df = load_data()

# pick a pollutant to explore
pollutant = st.selectbox("Pick a pollutant", ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL'])

# avg levels by station
st.subheader(f"Avg {pollutant} levels by Station")
avg_data = df.groupby("id")[pollutant].mean().reset_index()
fig1 = px.bar(avg_data, x="id", y=pollutant, title=f"Average {pollutant} across Stations")
st.plotly_chart(fig1)

# trend over time
st.subheader(f"{pollutant} trend over time")
fig2 = px.line(df, x="date", y=pollutant, color="id", title=f"{pollutant} trend by station")
st.plotly_chart(fig2)
