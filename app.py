import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load Models
@st.cache_resource
def load_models():
    if not os.path.exists('random_forest_reg.pkl') or not os.path.exists('random_forest_clf.pkl'):
        st.error("Model files not found. Please ensure 'random_forest_reg.pkl' and 'random_forest_clf.pkl' are uploaded to the project directory.")
        st.stop()
    reg_model = joblib.load('random_forest_reg.pkl')
    clf_model = joblib.load('random_forest_clf.pkl')
    return reg_model, clf_model

reg_model, clf_model = load_models()

# Load dataset to get columns and medians
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_dataset.csv')
    return df

df = load_data()
model_columns = [col for col in df.columns if col not in ['Price_in_Lakhs', 'Price_per_SqFt', 'Good_Investment']]

# Extract options from columns
cities = sorted([col.replace('City_', '') for col in model_columns if col.startswith('City_')])
property_types = sorted([col.replace('Property_Type_', '') for col in model_columns if col.startswith('Property_Type_')])
furnished_statuses = sorted([col.replace('Furnished_Status_', '') for col in model_columns if col.startswith('Furnished_Status_')])
amenities_options = sorted([col.replace('Amenities_', '') for col in model_columns if col.startswith('Amenities_')])

# Get individual amenities for checkboxes
all_amenities = set()
for opt in amenities_options:
    all_amenities.update(opt.split(', '))
individual_amenities = sorted(list(all_amenities))

# Medians for missing numeric
medians = df[['BHK', 'Size_in_SqFt', 'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals']].median()

def preprocess_input(city, bhk, size, year_built, property_type, furnished_status, parking, security, amenities):
    # Initialize all columns to 0
    input_data = {col: 0 for col in model_columns}

    # Set known values
    input_data['BHK'] = bhk
    input_data['Size_in_SqFt'] = size
    input_data['Year_Built'] = year_built
    input_data['Floor_No'] = medians['Floor_No']
    input_data['Total_Floors'] = medians['Total_Floors']
    input_data['Age_of_Property'] = medians['Age_of_Property']
    input_data['Nearby_Schools'] = medians['Nearby_Schools']
    input_data['Nearby_Hospitals'] = medians['Nearby_Hospitals']

    # One-hot for categorical
    input_data[f'City_{city}'] = 1
    input_data[f'Property_Type_{property_type}'] = 1
    input_data[f'Furnished_Status_{furnished_status}'] = 1
    input_data['Parking_Space_Yes'] = 1 if parking == 'Yes' else 0
    input_data['Security_Yes'] = 1 if security == 'Yes' else 0
    amenities_str = ', '.join(sorted(amenities))
    if f'Amenities_{amenities_str}' in input_data:
        input_data[f'Amenities_{amenities_str}'] = 1

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    return input_df

# Streamlit UI
st.title("Smart Property Price Predictor")
st.markdown("Predict future property prices and investment potential using AI models.")

# Inputs
city = st.selectbox("City", ["Select a City..."] + cities, index=0)
bhk = st.number_input("BHK", max_value=10)
size = st.number_input("Size (Sq Ft)", max_value=10000)
year_built = st.number_input("Year Built", max_value=2023)
property_type = st.selectbox("Property Type", ["Select Property Type..."] + property_types, index=0)
furnished_status = st.selectbox("Furnished Status", ["Select Furnished Status..."] + furnished_statuses, index=0)
parking = st.selectbox("Parking", ["Select...", "Yes", "No"], index=0)
security = st.selectbox("Security", ["Select...", "Yes", "No"], index=0)

# Amenities checkboxes
st.subheader("Amenities")
selected_amenities = []
for amenity in individual_amenities:
    if st.checkbox(amenity):
        selected_amenities.append(amenity)

if st.button("Predict"):
    # Validation
    if city == "Select a City..." or property_type == "Select Property Type..." or furnished_status == "Select Furnished Status..." or parking == "Select..." or security == "Select..." or bhk == 0 or size == 0 or year_built == 0:
        st.error("Please fill in all required fields.")
    else:
        # Preprocess
        input_df = preprocess_input(city, bhk, size, year_built, property_type, furnished_status, parking, security, selected_amenities)

        # Predictions
        future_price = reg_model.predict(input_df)[0]
        investment_prob = clf_model.predict_proba(input_df)[0][1]
        good_investment = "Yes" if investment_prob > 0.5 else "No"

        # Display
        st.subheader("Prediction Results")
        st.write(f"**Future Price after 5 years:** ₹{future_price:,.2f} Lakhs")
        st.write(f"**Good Investment?** {good_investment}")
        st.write(f"**Probability Score:** {investment_prob:.2%}")

        if good_investment == "Yes":
            st.success("This property is a good investment!")
        else:
            st.error("This property may not be a good investment.")

# Optional: Feature Importance (for regression model)
if st.checkbox("Show Feature Importance"):
    importances = reg_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': model_columns, 'Importance': importances}).sort_values('Importance', ascending=False)
    st.bar_chart(feature_importance_df.set_index('Feature'))

# Optional: Basic Dataset Statistics
if st.checkbox("Show Dataset Statistics"):
    st.write(df.describe())

# Optional: Graphs
if st.checkbox("Show Graphs"):
    st.subheader("Price Distribution")
    st.bar_chart(df['Price_in_Lakhs'])

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=[np.number]).corr()
    st.write(corr.style.background_gradient(cmap='coolwarm'))
