import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained k-NN model and scaler
model = joblib.load('admission_probability_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit UI
st.title("College Admission Probability Prediction")

# Input for user to enter data
all_india_rank = st.number_input("Enter All India Rank:", min_value=1, max_value=1000, step=1)
nirf_ranking = st.number_input("Enter NIRF Ranking:", min_value=1, max_value=30, step=1)
placement_percentage = st.number_input("Enter Placement Percentage:", min_value=10, max_value=100, step=1)
median_placement_package = st.number_input("Enter Median Placement Package (in lakhs):", min_value=10, max_value=200, step=1)
distance_from_college = st.number_input("Enter Distance from College (in km):", min_value=10, max_value=2000, step=10)
government_funded = st.checkbox("Is the College Government Funded?")
teachers_qualification = st.number_input("Enter Teachers Qualification (1 to 10):", min_value=1, max_value=10, step=1)
college_fee = st.number_input("Enter College Fee (in lakhs):", min_value=1, max_value=10, step=1)
living_facilities = st.number_input("Enter Living Facilities (1 to 10):", min_value=1, max_value=10, step=1)
girls_boys_ratio_percentage = st.number_input("Enter Girls/Boys Ratio Percentage (1 to 100):", min_value=1, max_value=100, step=1)

# Create a DataFrame with the input data
new_data = {
    'All_India_Rank': [all_india_rank],
    'NIRF_Ranking': [nirf_ranking],
    'Placement_Percentage': [placement_percentage],
    'Median_Placement_Package': [median_placement_package],
    'Distance_from_College': [distance_from_college],
    'Government_Funded': [1 if government_funded else 0],
    'Teachers_Qualification': [teachers_qualification],
    'College_Fee': [college_fee],
    'Living_Facilities': [living_facilities],
    'Girls_Boys_Ratio_Percentage': [girls_boys_ratio_percentage],
}

input_df = pd.DataFrame(new_data)

# Standardize the features using the same scaler used during training
new_data_scaled = scaler.transform(input_df)

# Button to trigger prediction
if st.button("Percentage Probability to Join the College"):
    # Make predictions using the trained model
    st.write(f"Input Features (scaled): {new_data_scaled}")
    predicted_admission_probability = model.predict(new_data_scaled)
    st.write(f'Predicted Admission Probability: {predicted_admission_probability[0]}%')
