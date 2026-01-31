import warnings
import streamlit as st
import pandas as pd
import joblib

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

# ----------------------------
# ALL amenities (MUST match training)
# ----------------------------
ALL_AMENITIES = [
    "Air Conditioning",
    "Wi-Fi",
    "Balcony",
    "Microwave",
    "Refrigerator",
    "TV",
    "Wardrobe",
    "Chandelier",
    "Dining Area",
    "24-Hour Electricity",
    "Apartment",
    "Dishwasher",
    "Hot Water",
    "Kitchen Cabinets",
    "Kitchen Shelf",
    "Pop Ceiling",
    "Pre-Paid Meter",
    "Tiled Floor"
]


# ----------------------------
# User inputs
# ----------------------------
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2)
floor_area = st.slider("Floor Area (sqm)", min_value=0, max_value=1400, value=50)

condition = st.selectbox("Condition", ["New", "Renovated", "Used"])
is_furnished = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
parking_space = st.checkbox("Parking Space Available")

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

locality = st.text_input("Locality (e.g., East Legon, Spintex)")

amenities_selected = st.multiselect(
    "Amenities",
    options=ALL_AMENITIES,
    default=[]
)

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
    "parking_space": [parking_space],
    "amenities": [[amenities_selected]]  # list of selected amenities
})



# -----------------------------
# Make prediction when button clicked
# -----------------------------
if st.button("Predict Rent"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Monthly Rent: GHS {prediction:,.2f}")

