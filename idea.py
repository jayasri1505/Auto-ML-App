import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming 'df' is your DataFrame and is already defined
df=pd.read_csv("dataset (1).csv")

st.title("Automated ML Model")
    
    # Selecting the target column
chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    # Creating a checklist for selecting independent variables
available_features = df.columns[df.columns != chosen_target].tolist()  # Exclude the target column
selected_features = st.multiselect('Select Independent Variables', available_features)

if st.button('Run Modelling'):
    if not selected_features:
        st.error("Please select at least one independent variable.")
    else:
        X = df[selected_features]  # Selected independent variables
        y = df[chosen_target]  # Target variable
            
        st.write(print(X))
