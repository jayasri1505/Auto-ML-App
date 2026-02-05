import pandas as pd
import streamlit as st
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer






st.title("Binary classification model")

# Title



with st.sidebar:
    st.image("file.png")
    st.title("autostreamml")
    choice=st.radio("Navigation",["upload","profiling","ml","download"])
    st.info("this application you to build a automate using auto ml")
    
# Necessary detial in side bar
    
    
    


if os.path.exists("sourcedata.csv"):
    df=pd.read_csv("sourcedata.csv",index_col=None)
    
# retriving source data
    

    
if choice == "upload":
    file=st.file_uploader("upload your Dataset Here")
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)
        
# uploading a csv file
        
        
if choice == "profiling":
    st.title("exploratory data analysis")
    Profile_report=ProfileReport(df)
    st_profile_report(Profile_report)
    
    
# profiling for the data analysist the data
    
if choice == "ml":
    st.title("automated ml model")
    
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    available_features = df.columns[df.columns != chosen_target].tolist()  # Exclude the target column
    selected_features = st.multiselect('Select Independent Variables', available_features)
    
    # selecting the target value
    
    if st.button('Run Modelling'):
        if not selected_features:
            st.error("Please select at least one independent variable.")
        else:
            X = df[selected_features]  # Selected independent variables
            y = df[chosen_target]  # Target variable
        # spliting dependent and independent columns
        
            if y.dtype == 'object':  # Check if target is categorical
                le = LabelEncoder()
                y = le.fit_transform(y)
            elif y.dtype == 'bool':  # Convert boolean target to integer (0, 1)
                y = y.astype(int)
            
         
         
        #spliting the data train test
        
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()
        
        #based on the catagory spliting the data    
         
         
        X[boolean_cols] = X[boolean_cols].astype(int)
         
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        
        # Preprocessing for numeric data (impute with median)
        numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
        ])
        
        
        
        
        # Preprocessing for categorical data (impute with most frequent and one-hot encode, with sparse_output=False)
        categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        
        # Bundle preprocessing for numeric and categorical data
        preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
        ])
        
        
        
        # Create a pipeline with preprocessing and TPOT classifier
        model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42))
        ])
        
        
        # Train the model pipeline
        model_pipeline.fit(X_train, y_train)
        
        accuracy = model_pipeline.score(X_test, y_test)
        
        
        st.write(f"Model Accuracy: {accuracy}")

        
        
        
        
        st.info("Training TPOT AutoML model, this might take a few minutes...")
        
        accuracy = model_pipeline.score(X_test, y_test)
        st.success(f"Model trained with Test Accuracy: {accuracy:.2%}")
        
        
        
        
        
        
        
        tpot_classifier = model_pipeline.named_steps['classifier']
        

        # Check if it is indeed a TPOTClassifier
        if isinstance(tpot_classifier, TPOTClassifier):
        # Export the best model pipeline
            tpot_classifier.export('best_model_pipeline.py')
        
    
        else:
            st.error("The model is not a TPOTClassifier.")
        
        
        
        
        
        
        
        
        
        
        
        st.info("The best model pipeline has been exported as 'best_model_pipeline.py'.")
        st.text("Optimized Model Pipeline: ")
        with open('best_model_pipeline.py', 'r') as file:
            st.code(file.read(), language='python')
            
            
if choice =="download":

    if os.path.exists('best_model_pipeline.py'):
        st.text("Optimized Model Pipeline: ")
        with open('best_model_pipeline.py', 'r') as file:
            st.code(file.read(), language='python')

        # Provide a download button
        with open('best_model_pipeline.py', 'rb') as f:
            st.download_button(
                label="Download Best Model Pipeline",
                data=f,
                file_name='best_model_pipeline.py',
                mime='text/x-python'
            )
    else:
        st.error("No pipeline has been created yet. Please run the model first.")
    
