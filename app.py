import streamlit as st
import joblib
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Churn Prediction")

# Load the pre-trained model
df_path = 'df.pkl'
pipeline_path = 'pipeline.joblib'

# Check if the files exist
if not os.path.exists(df_path) or not os.path.exists(pipeline_path):
    st.error("Required files not found. Please ensure 'df.pkl' and 'pipeline.joblib' are present.")
else:
    with open(df_path, 'rb') as file:
        df = pickle.load(file)

    # Load the pipeline using joblib
    pipeline = joblib.load(pipeline_path)

    # Streamlit app
    st.title("Customer Churn Prediction")

    # Create input fields for each feature in the dataframe
    st.header('Customer Data Input')
    input_data = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            input_data[column] = st.text_input(f"Enter {column}")
        elif column in ['HasCrCard', 'IsActiveMember']:  # Columns that should only accept 0 or 1
            input_data[column] = st.selectbox(f"Enter {column}", options=["Yes", "No"])
        elif df[column].dtype == 'int64':
            input_data[column] = st.number_input(
                f"Enter {column}", 
                value=int(df[column].median()), 
                step=1
            )
        elif df[column].dtype == 'float64':
            input_data[column] = st.number_input(
                f"Enter {column}", 
                value=float(df[column].median())
            )

    # Convert 'Yes'/'No' to 1/0 for HasCrCard and IsActiveMember
    input_data['HasCrCard'] = 1 if input_data['HasCrCard'] == "Yes" else 0
    input_data['IsActiveMember'] = 1 if input_data['IsActiveMember'] == "Yes" else 0

    # Convert input data to dataframe
    input_df = pd.DataFrame([input_data])

    # Predict churn
    if st.button("Predict"):
        prediction = pipeline.predict(input_df)
        st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")

    # Optionally display the input data for verification
    st.write("Input Data")
    st.write(input_df)
