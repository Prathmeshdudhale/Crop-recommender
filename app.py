import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('dataset.csv')

# Define the feature columns (exclude 'crop' and any irrelevant columns)
feature_columns = ['is_rain_falling', 'is_drought', 'soil_moisture', 'market_demand', 'soil_type', 'season']

# Title for the Streamlit app
st.title("🌾 Crop Prediction using Cosine Similarity 🌾")

# Introduction about the app
st.write("""
Welcome to the **Crop Prediction App** built by **Me**! This app helps farmers and agriculturists predict the best crop to grow based on various environmental and market factors. The app uses **cosine similarity** to find the crop that closely matches the given input conditions.
I have worked hard to bring this useful tool to life, and it's part of the exciting projects he's working on with a friend who's also a computer science student.
""")

# Input fields for the user to select values
st.subheader("Input the following conditions:")
is_rain_falling = st.selectbox("Is rain falling?", df['is_rain_falling'].unique())
is_drought = st.selectbox("Is there a drought?", df['is_drought'].unique())
soil_moisture = st.selectbox("Soil moisture level", df['soil_moisture'].unique())
market_demand = st.selectbox("Market demand", df['market_demand'].unique())
soil_type = st.selectbox("Soil type", df['soil_type'].unique())
season = st.selectbox("Season", df['season'].unique())

# Collect user input into a dictionary
input_data = {
    'is_rain_falling': is_rain_falling,
    'is_drought': is_drought,
    'soil_moisture': soil_moisture,
    'market_demand': market_demand,
    'soil_type': soil_type,
    'season': season
}

# Convert the input to a DataFrame
input_df = pd.DataFrame([input_data])

# Apply Label Encoding to both the dataset and the input
label_encoders = {}
for column in feature_columns + ['crop']:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Apply label encoding to input data
for column in input_df.columns:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Convert input_df and relevant data for cosine similarity
input_array = input_df.values
data_array = df[feature_columns].values  # Use only the feature columns, not 'crop'

# Calculate cosine similarity
cosine_sim = cosine_similarity(input_array, data_array)

# Find the most similar data point
most_similar_index = np.argmax(cosine_sim)

# Predict the crop
predicted_crop = label_encoders['crop'].inverse_transform([df['crop'][most_similar_index]])

# Display the predicted crop
# Display the predicted crop with color highlighting and larger font size
predicted_crop_text = f"""
The predicted crop based on the given conditions is: <span style='color:green; font-size:24px; font-weight:bold;'>{predicted_crop[0]}</span>
"""
st.markdown(predicted_crop_text, unsafe_allow_html=True)

# Footer section acknowledging Prathmesh in the bottom-right corner
footer = """
    <style>
    .footer {
        position: fixed;
        right: 10px;
        bottom: 10px;
        font-size: 14px;
        color: grey;
    }
    </style>
    <div class="footer">
        Made By Prathmesh ❤️
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)

