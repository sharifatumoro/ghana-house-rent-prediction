import streamlit as st
import pandas as pd
import joblib
import json

from pathlib import Path



# --------------------------------
# Load trained model
# --------------------------------

model = joblib.load(
    r"C:\Users\name\Desktop\BIG FOLDER\PROJECTS\My_Projects\ghana_house_rent_project\model\final_model.pkl"
)


# --------------------------------
# Load localities
# --------------------------------

with open(r"C:\Users\name\Desktop\ghana-house-rent-prediction\data\localities.json", encoding="utf-8") as f:
    LOCALITIES = json.load(f)   # now a dict: {region: [localities]}


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Ghana House Rent Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Ghana House Rent Prediction App")

st.write(
    "Enter the property details below to estimate its monthly rental price in Ghana."
)


# --------------------------------
# User inputs
# --------------------------------

st.subheader("Property Details")


# Numerical features

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=0,
    max_value=20,
    value=2
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=0,
    max_value=20,
    value=2
)

floor_area = st.number_input(
    "Floor Area (sqm)",
    min_value=0,
    max_value=1400,
    value=50, 
    step = 10
)


# Ordinal feature

condition = st.selectbox(
    "Property Condition",
    ["New", "Renovated", "Used"]
)


# Categorical features

is_furnished = st.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)

category = st.selectbox(
    "Property Category",
    [
        "Flats",
        "Detached",
        "Townhouse",
        "Duplex",
        "Mansion",
        "Semi-Detached"
    ]
)


# --------------------------------
# Region and Locality
# --------------------------------

st.subheader("Location Details")

region = st.selectbox(
    "Select Region",
    sorted(LOCALITIES.keys())
)

locality = st.selectbox(
    "Select Locality",
    LOCALITIES[region]
)


# --------------------------------
# Grouped Amenity Features
# --------------------------------

st.subheader("Property Features and Amenities")

# Utilities
with st.expander("⚡ Utilities"):
    st.caption('Basic services and connections available in the property. E.g., 24-Hour Electricity, Pre-Paid Meter, Wi-Fi.')

    utilities = st.checkbox("Utilities available")


# Kitchen Facilities
with st.expander("🍳 Kitchen Facilities"):
    st.caption('Features and appliances that support food preparation and storage. E.g., Kitchen Cabinets, Kitchen Shelf, Dishwasher, Microwave, Refrigerator.')

    kitchen_facilities = st.checkbox("Kitchen facilities available")


# Comfort
with st.expander("❄️ Comfort"):
    st.caption('Features that improve the comfort and convenience of occupants. E.g., Air Conditioning, Hot Water.')

    comfort = st.checkbox("Comfort features available")


# Interior Finishing
with st.expander("✨ Interior Finishing"):
    st.caption("Decorative and finishing features that enhance the property's appearance. E.g., Tiled Floor, Pop Ceiling, Chandelier.")

    interior_finishing = st.checkbox("Interior finishing available")


# Living Space
with st.expander("🏡 Living Space"):
    st.caption('Additional spaces that improve the functionality of the property. E.g., Dining Area, Balcony.')

    living_space = st.checkbox("Living space features available")


# Wardrobe
with st.expander("👔 Wardrobe"):
    st.caption("Storage features available in the property. E.g., Built-in Wardrobe, Walk-in Wardrobe.")

    wardrobe = st.checkbox("Wardrobe available")


# Television
with st.expander("📺 Television"):
    st.caption("Entertainment facilities available in the property. E.g., Television, Smart TV.")

    tv = st.checkbox("Television available")
    

# --------------------------------
# Prepare input for prediction
# --------------------------------

input_data = pd.DataFrame({

    "bedrooms": [bedrooms],

    "bathrooms": [bathrooms],

    "floor_area": [floor_area],

    "condition": [condition],

    "is_furnished": [is_furnished],

    "region": [region],

    "category": [category],

    "locality": [locality],

    "utilities": [int(utilities)],

    "kitchen_facilities": [int(kitchen_facilities)],

    "comfort": [int(comfort)],

    "interior_finishing": [int(interior_finishing)],

    "living_space": [int(living_space)],

    "wardrobe": [int(wardrobe)],

    "tv": [int(tv)]

})


# --------------------------------
# Make Prediction
# --------------------------------

st.divider()

if st.button("Predict Rent", use_container_width=True):

    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Estimated Monthly Rent: GHS {prediction:,.2f}"
    )