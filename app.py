import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved Random Forest model
model, feature_names = joblib.load('luxury_model.pkl')

st.title("🏡 Luxury Property Prediction App")
st.write("Enter the property details below to check if it's likely a **Luxury Property** or not.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Property Info")
    price = st.number_input("Price ($)", min_value=0.0, value=300000.0)
    square_footage = st.number_input("Square Footage", min_value=0.0, value=2000.0)
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=2.0)
    backyard_space = st.number_input("Backyard Space (sq ft)", min_value=0.0, value=500.0)
    age_of_home = st.number_input("Age of Home (years)", min_value=0.0, value=10.0)

with col2:
    st.subheader("Location & Quality")
    crime_rate = st.number_input("Crime Rate", min_value=0.0, value=15.0)
    school_rating = st.slider("School Rating (1–10)", 1, 10, 7)
    distance_to_city = st.number_input("Distance to City Center (km)", min_value=0.0, value=5.0)
    employment_rate = st.number_input("Employment Rate (%)", min_value=0.0, max_value=100.0, value=85.0)
    property_tax_rate = st.number_input("Property Tax Rate (%)", min_value=0.0, value=1.5)
    local_amenities = st.slider("Local Amenities (1–10)", 1, 10, 7)
    transport_access = st.slider("Transport Access (1–10)", 1, 10, 7)
    renovation_quality = st.slider("Renovation Quality (1–10)", 1, 10, 7)

# Create input array for prediction - MUST match the exact order of features the model expects
# Order: ['SquareFootage', 'NumBathrooms', 'BackyardSpace', 'NumBedrooms', 'CrimeRate', 
#         'DistanceToCityCenter', 'SchoolRating', 'AgeOfHome', 'EmploymentRate', 
#         'PropertyTaxRate', 'LocalAmenities', 'TransportAccess', 'RenovationQuality', 'Price']
features = np.array([[
    square_footage,      # SquareFootage
    num_bathrooms,       # NumBathrooms
    backyard_space,      # BackyardSpace
    num_bedrooms,        # NumBedrooms
    crime_rate,          # CrimeRate
    distance_to_city,    # DistanceToCityCenter
    school_rating,       # SchoolRating
    age_of_home,         # AgeOfHome
    employment_rate,     # EmploymentRate
    property_tax_rate,   # PropertyTaxRate
    local_amenities,     # LocalAmenities
    transport_access,    # TransportAccess
    renovation_quality,  # RenovationQuality
    price                # Price
]])

# Predict
if st.button("Predict Luxury Status"):
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.success(f"💎 This property is **Luxury** (Confidence: {probability:.2%})")
        else:
            st.warning(f"🏠 This property is **Not Luxury** (Confidence: {probability:.2%})")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that all input fields are filled correctly.")
