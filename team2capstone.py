import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

# Load your dataset (assuming it's in a DataFrame format)
# Replace 'your_dataset.csv' with the actual path to your dataset

@st.cache
def load_data():
    data = pd.read_csv('disease prediction data.csv')
    return data

# Define treatment recommendations dictionary
treatment_recommendations = {
    'Circulatory system diseases': ['Potassium Chloride', 'D5W', 'Furosemide', 'Insulin', 'Metoprolol', 'Acetaminophen', 'Metoprolol Tartrate', 'Sodium Chloride 0.9%  Flush', 'NS', '0.9% Sodium Chloride'],
    'Complications of pregnancy, childbirth, and the puerperium': ['NS', 'Potassium Chloride', 'D5W', 'Insulin', 'SW', 'Magnesium Sulfate', 'Furosemide', 'Acetaminophen', 'Sodium Chloride 0.9%  Flush', 'Iso-Osmotic Dextrose'],
    'Congenital anomalies': ['Potassium Chloride', 'D5W', 'Furosemide', 'Acetaminophen', 'HYDROmorphone (Dilaudid)', '0.9% Sodium Chloride', 'Insulin', 'Magnesium Sulfate', 'NS', 'Morphine Sulfate'],
    'Digestive system diseases': ['Potassium Chloride', '0.9% Sodium Chloride', 'NS', 'Furosemide', 'Magnesium Sulfate', 'Insulin', 'Sodium Chloride 0.9%  Flush', 'D5W', 'SW', '5% Dextrose'],
    'Diseases of the blood and blood-forming organs': ['0.9% Sodium Chloride', 'NS', 'Potassium Chloride', 'Insulin', 'Furosemide', 'Magnesium Sulfate', 'Acetaminophen', '5% Dextrose', 'Sodium Chloride 0.9%  Flush', 'Iso-Osmotic Dextrose'],
    'Endocrine, nutritional and metabolic diseases, and immunity disorders': ['Insulin', 'Potassium Chloride', 'NS', '0.9% Sodium Chloride', 'Magnesium Sulfate', 'Sodium Chloride 0.9%  Flush', 'D5W', 'Heparin', 'Iso-Osmotic Dextrose', 'Furosemide'],
    'External causes of injury and supplemental classification': ['D10W', 'Syringe (Neonatal) *D5W*', 'Potassium Chloride', 'Sodium Chloride', 'Syringe (Neonatal)', 'NEO*IV*Gentamicin', 'Heparin (Preservative Free)', 'Send 500mg Vial', 'NEO*PO*Ferrous Sulfate Elixir', 'NEO*IV*Caffeine Citrate'],
    'Genitourinary system diseases': ['Potassium Chloride', '0.9% Sodium Chloride', 'NS', 'Iso-Osmotic Dextrose', 'HYDROmorphone (Dilaudid)', 'Magnesium Sulfate', 'Insulin', 'Sodium Chloride 0.9%  Flush', 'D5W', 'SW'],
    'Ill-defined conditions': ['Potassium Chloride', '0.9% Sodium Chloride', 'NS', 'Insulin', 'Sodium Chloride 0.9%  Flush', 'D5W', 'Magnesium Sulfate', 'Lorazepam', 'Iso-Osmotic Dextrose', 'Acetaminophen'],
    'Infectious and parasitic diseases': ['Potassium Chloride', 'D5W', 'NS', '0.9% Sodium Chloride', 'Insulin', 'Furosemide', 'Iso-Osmotic Dextrose', 'Magnesium Sulfate', 'Sodium Chloride 0.9%  Flush', '5% Dextrose'],
    'Injury and poisoning': ['Potassium Chloride', '0.9% Sodium Chloride', 'NS', 'Insulin', 'D5W', 'Magnesium Sulfate', 'Furosemide', 'Iso-Osmotic Dextrose', 'Sodium Chloride 0.9%  Flush', 'SW'],
    'Mental disorders': ['Diazepam', 'Potassium Chloride', '0.9% Sodium Chloride', 'Magnesium Sulfate', 'Sodium Chloride 0.9%  Flush', 'Lorazepam', 'NS', 'FoLIC Acid', 'Thiamine', 'Heparin'],
    'Musculoskeletal system and connective tissue diseases': ['Potassium Chloride', 'Insulin', 'NS', '0.9% Sodium Chloride', 'Iso-Osmotic Dextrose', 'HYDROmorphone (Dilaudid)', 'Sodium Chloride 0.9%  Flush', 'Magnesium Sulfate', 'D5W', 'Furosemide'],
    'Neoplasms': ['0.9% Sodium Chloride', 'NS', 'Potassium Chloride', 'Furosemide', 'D5W', '5% Dextrose', 'Lorazepam', 'Insulin', 'SW', 'Magnesium Sulfate'],
    'Nervous system diseases and sense organs': ['0.9% Sodium Chloride', 'Potassium Chloride', 'Lorazepam', 'Insulin', 'Sodium Chloride 0.9%  Flush', 'NS', 'Magnesium Sulfate', 'Iso-Osmotic Dextrose', '5% Dextrose', 'Acetaminophen'],
    'Respiratory system diseases': ['Potassium Chloride', 'Insulin', 'Furosemide', 'NS', 'D5W', '0.9% Sodium Chloride', 'Iso-Osmotic Dextrose', 'Magnesium Sulfate', 'SW', 'Sodium Chloride 0.9%  Flush'],
    'Skin and subcutaneous tissue diseases': ['0.9% Sodium Chloride', 'Potassium Chloride', '5% Dextrose', 'D5W', 'NS', 'Magnesium Sulfate', 'SW', 'Furosemide', 'Iso-Osmotic Dextrose', 'Sodium Chloride 0.9%  Flush']
}

# Define the main function
def main():
    st.title("Disease Prediction with Random Forest Classifier")

    # Load data
    data = load_data()

    # Remove HADM_ID column from features and set Disease_Name as the target variable
    X = data.drop(columns=['HADM_ID', 'Disease_Name', 'SUBJECT_ID'])
    y = data['Disease_Name']

    # Display some information about the dataset
    st.sidebar.header("Dataset Information")
    st.sidebar.write("Number of samples:", X.shape[0])
    st.sidebar.write("Number of features:", X.shape[1])
    st.sidebar.write("Number of unique diseases:", len(y.unique()))

    # Allow user to input data for prediction
    st.sidebar.header("Input Data for Prediction")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.sidebar.text_input(f"{feature}:", value=X[feature].mean())

    # Create DataFrame from user input data
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = predict_disease(input_df, X, y)

    # Display prediction
    st.subheader("Prediction:")
    st.write("Predicted Disease:", prediction[0])

    # Display treatment recommendation
    get_treatment_recommendation(prediction[0])

# Function to predict disease
def predict_disease(input_df, X, y):
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    rf_classifier = RandomForestClassifier(class_weight='balanced')
    rf_classifier.fit(X_resampled, y_resampled)

    # Make prediction
    prediction = rf_classifier.predict(input_df)

    return prediction

# Function to get treatment recommendation
def get_treatment_recommendation(prediction):
    if prediction in treatment_recommendations:
        st.subheader("Treatment Recommendation:")
        st.write("Recommended Treatments:", treatment_recommendations[prediction])
    else:
        st.subheader("Treatment Recommendation:")
        st.write("No treatment recommendation available for the predicted disease.")

if __name__ == '__main__':
    main()
