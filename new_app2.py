#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Load the machine learning model
model1 = joblib.load('best_rf_classifier_dummy.pkl')

# Load scaler
loaded_scaler = joblib.load('scaler_dummy.save')

# Define the main function to create the Streamlit app
def main():
    # Set the title of the Streamlit app
    st.title('RISK STRATIFICATION')
    age = st.number_input('Age', min_value=0, max_value=120, value=40)
    heart_rate = st.number_input('HEART_RATE',min_value = 0, max_value=200, value=100)
    SPO2 = st.number_input('SPO2',min_value = 0, max_value=100, value=98)
    HEMOGLOBIN = st.number_input('HEMOGLOBIN',min_value = 0, max_value=22, value=11)
    BP_dia = st.number_input('BP_dia',min_value = 0, max_value=100, value=68)
    BP_sys = st.number_input('BP_sys',min_value = 0, max_value=240, value=128)
    Temp_F = st.number_input('Temp_F',min_value = 0, max_value=104, value=98)
    Admit_wt = st.number_input('Admit_wt',min_value = 0, max_value=472, value=86)
    Height = st.number_input('Height',min_value = 0, max_value=175, value=68)
    LOS = st.number_input('LOS',min_value = 0, max_value=260, value=5)
    ADMISSION_TYPE = st.selectbox('ADMISSION_TYPE',['EMERGENCY', 'NEWBORN', 'ELECTIVE', 'URGENT'])
    GENDER = st.selectbox('GENDER',['Male', 'Female'])
    curr_service = st.selectbox('Curr_Service',['CMED', 'NB', 'NBB', 'MED', 'SURG', 'CSURG', 'NMED', 'TRAUM','VSURG', 'PSURG', 'NSURG', 'ORTHO', 'GU', 'ENT', 'OMED', 'TSURG','GYN', 'OBS', 'DENT'])
    FIRST_CAREUNIT = st.selectbox('FIRST_CAREUNIT',['MICU', 'CCU', 'NICU', 'TSICU', 'SICU', 'CSRU'])
    ICD9_CODE = st.selectbox('disease_diagnosis',['Infectious and Parasitic Diseases','Neoplasms','Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders','Diseases of the Blood and Blood-forming Organs','Mental Disorders','Diseases of the Nervous System and Sense Organs','Diseases of the Circulatory System','Diseases of the Respiratory System','Diseases of the Digestive System','Diseases of the Genitourinary System','Complications of Pregnancy, Childbirth, and the Puerperium','Diseases of the Skin and Subcutaneous Tissue','Diseases of the Musculoskeletal System and Connective Tissue','Congenital Anomalies','Certain Conditions originating in the Perinatal Period','Symptoms, Signs and Ill-defined Conditions','Injury and Poisoning','Supplementary Classification of External Causes of Injury and Poisoning','Supplementary Classification of Factors influencing Health Status and Contact with Health Services'])

    if st.button('Calculate Probability by Random Forest'):
        try:
            lis = [age, heart_rate, SPO2, HEMOGLOBIN, BP_dia, BP_sys, Temp_F, Admit_wt, Height, LOS]
            if ADMISSION_TYPE == 'ELECTIVE':
                lis.extend([1, 0, 0, 0])
            elif ADMISSION_TYPE == 'EMERGENCY':
                lis.extend([0, 1, 0, 0])
            elif ADMISSION_TYPE == 'NEWBORN':
                lis.extend([0, 0, 1, 0])
            elif ADMISSION_TYPE == 'URGENT':
                lis.extend([0, 0, 0, 1])

            if GENDER == 'Female':
                lis.extend([1, 0])
            elif GENDER == 'Male':
                lis.extend([0, 1])

            if curr_service:
                services = ['CMED', 'CSURG', 'DENT', 'ENT', 'GU', 'GYN', 'MED', 'NB', 'NBB', 'NMED', 'NSURG', 'OBS', 'OMED', 'ORTHO', 'PSURG', 'SURG', 'TRAUM', 'TSURG', 'VSURG']
                for service in services:
                    if curr_service == service:
                        lis.extend([1])
                    else:
                        lis.extend([0])

            if FIRST_CAREUNIT:
                units = ['CCU', 'CSRU', 'MICU', 'NICU', 'SICU', 'TSICU']
                for unit in units:
                    if FIRST_CAREUNIT == unit:
                        lis.extend([1])
                    else:
                        lis.extend([0])

            if ICD9_CODE:
                diseases = ['Infectious and Parasitic Diseases', 'Neoplasms', 'Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders', 'Diseases of the Blood and Blood-forming Organs', 'Mental Disorders', 'Diseases of the Nervous System and Sense Organs', 'Diseases of the Circulatory System', 'Diseases of the Respiratory System', 'Diseases of the Digestive System', 'Diseases of the Genitourinary System', 'Complications of Pregnancy, Childbirth, and the Puerperium', 'Diseases of the Skin and Subcutaneous Tissue', 'Diseases of the Musculoskeletal System and Connective Tissue', 'Congenital Anomalies', 'Certain Conditions originating in the Perinatal Period', 'Symptoms, Signs and Ill-defined Conditions', 'Injury and Poisoning', 'Supplementary Classification of External Causes of Injury and Poisoning', 'Supplementary Classification of Factors influencing Health Status and Contact with Health Services']
                for disease in diseases:
                    if ICD9_CODE == disease:
                        lis.extend([1])
                    else:
                        lis.extend([0])

            lis = np.array(lis).reshape(1, -1)

            # Scale the features
            scaled_features = loaded_scaler.transform(lis)

            # Predict probability
            probability = model1.predict_proba(scaled_features)[:, 1]
            st.write(f"Probability of expiry: {probability[0]}")
            
            # Determine risk level
            if probability < 0.25:
                risk_level = "Low"
            elif probability < 0.66:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            st.write(f"Risk Level: {risk_level}")
            
            # Create a gauge chart
            create_gauge_chart(probability[0], risk_level)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def create_gauge_chart(probability, risk_level):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Define colors
    low_color = '#7FFF7F'  # light green
    moderate_color = '#FFFF7F'  # light yellow
    high_color = '#FF7F7F'  # light red

    # Draw gauge sectors
    ax.add_patch(Arc((0.5, 0.8), 0.8, 0.4, theta1=120, theta2=180, color=low_color, linewidth=2))
    ax.add_patch(Arc((0.5, 0.8), 0.8, 0.4, theta1=45, theta2=120, color=moderate_color, linewidth=2))
    ax.add_patch(Arc((0.5, 0.8), 0.8, 0.4, theta1=0, theta2=45, color=high_color, linewidth=2))

    # Draw needle
# Draw needle
    angle = 180 - (180 * probability)
    ax.plot([0.5, 0.5 + 0.4 * np.cos(np.radians(angle))],[0.8, 0.8 + 0.4 * np.sin(np.radians(angle))], color='black', linewidth=3)


    # Set axis limits and remove ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Display risk level
    plt.text(0.5, 0.7, risk_level, horizontalalignment='center', verticalalignment='center', fontsize=14, fontweight='bold')

    plt.title('Risk Level')
    st.pyplot(fig)

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
