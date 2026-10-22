import warnings
import streamlit as st
import pandas as pd
import joblib
import json

# Suppress scikit-learn version warnings
# -----------------------------
warnings.filterwarnings("ignore", message="to unpickle estimator Lasso from version")

# MUST import encoder BEFORE loading model
from encoders.multihotencoder import MultiHotEncoder

# ----------------------------
# Load model
# ----------------------------
model = joblib.load(
    r"C:\Users\name\Desktop\BIG FOLDER\PROJECTS\My_Projects\ghana_house_rent_project\model\final_model.pkl"
)

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Ghana House Rent Predictor",
    layout="centered"
)

st.title("🏠 Ghana House Rent Prediction App")
st.write("Enter property details below to estimate monthly rent in Ghana")


with open(r"C:\Users\name\Desktop\BIG FOLDER\PROJECTS\My_Projects\ghana_house_rent_project\model\localities.json") as f:
    LOCALITIES = json.load(f)

pre = model.named_steps['preprocessor']

# Amenities options, taken from the trained encoder
multihot = pre.named_transformers_['multihot']
ALL_AMENITIES = list(multihot.mlb.classes_)

# ----------------------------
# User inputs
# ----------------------------
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2)
floor_area = st.slider("Floor Area (sqm)", min_value=0, max_value=1400, value=50)

condition = st.selectbox("Condition", ["New", "Renovated", "Used"])
is_furnished = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])

category = st.selectbox(
    "Property Category",
    ["Flats", "Detached", "Townhouse", "Duplex", "Mansion", "Semi-Detached"]
)

region = st.selectbox(
    "Region",
    [
        "Greater Accra",
        "Ashanti",
        "Eastern Region",
        "Central Region",
        "Northern Region",
        "Western Region",
        "Brong Ahafo",
        "Volta Region"
    ]
)

locality = st.selectbox("Locality", LOCALITIES)

amenities_selected = st.multiselect("Amenities", ALL_AMENITIES)

# -----------------------------
# Prepare input for pipeline
# -----------------------------
# IMPORTANT: pass amenities as a list, pipeline will encode automatically
input_data = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "floor_area": [floor_area],
    "condition": [condition],
    "is_furnished": [is_furnished],
    "locality": [locality],
    "region": [region],
    "category": [category],
    "amenities": [amenities_selected] # list of selected amenities
})



# -----------------------------
# Make prediction when button clicked
# -----------------------------
if st.button("Predict Rent"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Monthly Rent: GHS {prediction:,.2f}")


