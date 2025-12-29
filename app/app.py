import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load(r"C:\Users\name\Desktop\PROJECTS\My_Projects\ghana_house_rent_project\model\final_model.pkl")
scaler = joblib.load(r"C:\Users\name\Desktop\PROJECTS\My_Projects\ghana_house_rent_project\model\scaler.pkl")
ordinal_mappings = joblib.load(r"C:\Users\name\Desktop\PROJECTS\My_Projects\ghana_house_rent_project\model\ordinal_encoder.pkl")
target_mean_mappings = joblib.load(r"C:\Users\name\Desktop\PROJECTS\My_Projects\ghana_house_rent_project\model\target_mean_encoder.pkl")

# Load your original dataset (so we get category names)
df = pd.read_csv(r"C:\Users\name\Desktop\PROJECTS\My_Projects\ghana_house_rent_project\data\cleaned_house_rentals.csv") 

# Fix Target Mean Mappings Using df
target_mean_mappings = {}
for col in ['category', 'amenities', 'locality']:
    target_mean_mappings[col] = df.groupby(col)['price'].mean().to_dict()

# Streamlit Layout
st.set_page_config(page_title=" Ghana House Rent Price Predictor", layout="centered")
st.title(" Ghana House Rent Price Predictor")
st.write("Enter property details below to estimate the monthly rent price in Ghana ")

# Sidebar Inputs
st.sidebar.header("Property Details")

# Use unique category names directly from df
location = st.sidebar.selectbox("Select Location", sorted(df['locality'].unique()))
property_type = st.sidebar.selectbox("Property Type", sorted(df['category'].unique()))
amenities = st.sidebar.selectbox("Amenities", sorted(df['amenities'].unique()))
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 8, 2)
furnished = st.sidebar.selectbox("Furnishing Level", list(ordinal_mappings['furnished_order'].keys()))

# User Input (Raw for display)
user_input_raw = pd.DataFrame({
    'locality': [location],
    'category': [property_type],
    'amenities': [amenities],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'furnished': [furnished]
})

st.write("### User Input (Before Encoding)")
st.dataframe(user_input_raw)

# Preprocessing for Model
user_input = user_input_raw.copy()

# Ordinal Encoding
user_input['furnished'] = user_input['furnished'].map(ordinal_mappings['furnished_order'])


# Target Mean Encoding (final fix) 
for col in ['category', 'amenities', 'locality']:
    mapping = target_mean_mappings[col]
    val = user_input[col].iloc[0]
    if val in mapping:
        user_input[col] = mapping[val]
    else:
        user_input[col] = np.mean(list(mapping.values()))

# Reorder columns to match training data
expected_cols = scaler.feature_names_in_   
user_input = user_input[expected_cols]

# Scale numerical features 
user_input_scaled = scaler.transform(user_input)

# Make Prediction
if st.button("💰 Predict Rent Price"):
    prediction = model.predict(user_input_scaled)
    st.success(f"Estimated Rent Price: GH₵ {prediction[0]:,.2f}")
