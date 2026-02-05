import pandas as pd
import streamlit as st
import os
import re
from ydata_profiling import ProfileReport
from tpot import TPOTClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="autostreamml", layout="wide")
st.title("Binary classification model")

PAGES = ["upload", "profiling", "ml", "download"]

# Persist navigation across reruns + allow deep linking via URL
query_page = st.query_params.get("page")
if "page" not in st.session_state:
    st.session_state.page = query_page if query_page in PAGES else "upload"
elif query_page in PAGES and query_page != st.session_state.page:
    st.session_state.page = query_page

def _sync_nav_to_url() -> None:
    st.session_state.page = st.session_state.nav
    st.query_params["page"] = st.session_state.page

# Title



with st.sidebar:
    st.image("file.png")
    st.title("autostreamml")
    choice = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.page),
        key="nav",
        on_change=_sync_nav_to_url,
    )
    st.info("this application you to build a automate using auto ml")
    
# Necessary detial in side bar
    
    
    


# Load dataset only when user uploads in the current session
if "df" not in st.session_state:
    st.session_state.df = None
if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False
    
# retriving source data
    

    
if choice == "upload":
    file=st.file_uploader("upload your Dataset Here")
    if file:
        st.session_state.df = pd.read_csv(file,index_col=None)
        st.session_state.df.to_csv("sourcedata.csv",index=None)
        st.session_state.data_uploaded = True
        st.success("Dataset uploaded successfully.")
        st.dataframe(st.session_state.df)
        
# uploading a csv file
        
        
if choice == "profiling":
    st.title("exploratory data analysis")
    if not st.session_state.data_uploaded or st.session_state.df is None or st.session_state.df.empty:
        st.warning("Upload a dataset in the Upload tab first.")
    else:
        with st.spinner("Generating profiling report..."):
            df = st.session_state.df
            Profile_report = ProfileReport(df, minimal=True)
            html_report = Profile_report.to_html()
            
            # Aggressively remove navigation elements using string manipulation
            
            # Remove hamburger menu buttons and navigation elements
            html_report = re.sub(r'<button[^>]*aria-label[^>]*menu[^>]*>.*?</button>', '', html_report, flags=re.IGNORECASE | re.DOTALL)
            html_report = re.sub(r'<button[^>]*class[^>]*hamburger[^>]*>.*?</button>', '', html_report, flags=re.IGNORECASE | re.DOTALL)
            html_report = re.sub(r'<button[^>]*class[^>]*menu[^>]*>.*?</button>', '', html_report, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove navigation sidebars and overlays
            html_report = re.sub(r'<nav[^>]*>.*?</nav>', '', html_report, flags=re.IGNORECASE | re.DOTALL)
            html_report = re.sub(r'<div[^>]*class[^>]*sidebar[^>]*>.*?</div>', '', html_report, flags=re.IGNORECASE | re.DOTALL)
            html_report = re.sub(r'<div[^>]*class[^>]*nav[^>]*>.*?</div>', '', html_report, flags=re.IGNORECASE | re.DOTALL)
            
            # Inject comprehensive CSS to hide any remaining navigation elements
            hide_nav_css = """
            <style>
                /* Hide all navigation and menu elements */
                button[aria-label*="menu"], button[aria-label*="Menu"],
                button[class*="hamburger"], button[class*="menu"],
                .navbar, .nav-menu, .hamburger-menu, .menu-button,
                .sidenav, .side-nav, .navigation-menu, .nav,
                [class*="nav-menu"], [class*="hamburger"], [class*="menu-btn"],
                [id*="nav-menu"], [id*="hamburger"], [id*="menu-button"],
                .overlay, .modal-nav, [class*="overlay-nav"],
                .sidebar, .side-bar, [class*="sidebar"],
                nav, header button, .header button {
                    display: none !important;
                    visibility: hidden !important;
                    opacity: 0 !important;
                    width: 0 !important;
                    height: 0 !important;
                    overflow: hidden !important;
                }
                /* Hide any button in header area */
                header button, .header button, .top-bar button {
                    display: none !important;
                }
            </style>
            """
            # Insert CSS in the head
            html_report = html_report.replace('<head>', f'<head>{hide_nav_css}')
            
            # Embed with sufficient height to display full report
            st.components.v1.html(html_report, height=1200, scrolling=True)
    
    
# profiling for the data analysist the data
    
if choice == "ml":
    st.title("automated ml model")
    if not st.session_state.data_uploaded or st.session_state.df is None or st.session_state.df.empty:
        st.warning("Upload a dataset in the Upload tab first.")
        st.stop()
    
    df = st.session_state.df
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
        
        
        st.subheader("Model experimentation")
        st.caption("We’ll try a few baseline models first and show their test accuracy.")

        baseline_models = {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
            "SVC (RBF)": SVC(),
            "KNN": KNeighborsClassifier(),
            "GaussianNB": GaussianNB(),
        }

        baseline_results = []
        best_name = None
        best_acc = -1.0
        best_pipeline = None

        for name, clf in baseline_models.items():
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            baseline_results.append({"model": name, "accuracy": acc})
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_pipeline = pipe

        results_df = pd.DataFrame(baseline_results).sort_values("accuracy", ascending=False)
        st.dataframe(results_df, width="stretch")
        st.success(f"Best baseline model: {best_name} (accuracy: {best_acc:.2%})")

        # Save best baseline pipeline for download
        joblib.dump(best_pipeline, "best_sklearn_pipeline.pkl")
        st.info("Saved best baseline pipeline as `best_sklearn_pipeline.pkl` (downloadable in the Download tab).")

        run_tpot = st.checkbox("Also run TPOT AutoML (slower, may find a better pipeline)", value=False)
        if not run_tpot:
            st.stop()

        # Create a pipeline with preprocessing and TPOT classifier
        model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # TPOT >= 1.x uses `verbose` and time-based limits (not `verbosity`/generations/population_size)
        ('classifier', TPOTClassifier(
            verbose=2,
            max_time_mins=5,
            max_eval_time_mins=2,
            n_jobs=-1,
            random_state=42,
        ))
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
    if not st.session_state.data_uploaded:
        st.warning("Upload a dataset in the Upload tab first (then train a model) to enable downloads.")
        st.stop()

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
        st.warning("No TPOT pipeline has been created yet. Run TPOT from the ML tab if you want a .py export.")

    if os.path.exists("best_sklearn_pipeline.pkl"):
        st.text("Best Baseline (scikit-learn) Pipeline:")
        with open("best_sklearn_pipeline.pkl", "rb") as f:
            st.download_button(
                label="Download Best Baseline Pipeline (.pkl)",
                data=f,
                file_name="best_sklearn_pipeline.pkl",
                mime="application/octet-stream",
            )
    else:
        st.warning("No baseline pipeline saved yet. Run baseline models from the ML tab first.")
    
