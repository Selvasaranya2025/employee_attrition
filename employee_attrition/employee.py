import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import pickle

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Analysis and Prediction Dashboard")

# --------------------------
# Load Data and Model
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv("employee_attrition/cleaned_employee_attrition (6).csv")

# Load the data and model
df = load_data()  # Looks for employee_attrition/cleaned_employee_attrition (6).csv
rf_model = load("employee_attrition/random_forest_model.joblib")
with open(r"c:\Users\LOQ\saran_guvi\employee_attrition\feature_columns2.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Ensure feature_cols is a list
if isinstance(feature_cols, pd.Index):
    feature_cols = feature_cols.tolist()
    

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["EDA & Insights", "Feature Importance", "Prediction"])

# --------------------------
# with tabs[0]:
st.header("Exploratory Data Analysis (EDA)")

# Use 2 columns for compact layout
col1, col2 = st.columns(2)

    # --- Attrition Count ---
with col1:
    st.subheader("Attrition Count")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.countplot(x='attrition', data=df, ax=ax)
    ax.set_title("Attrition (0 = Stay, 1 = Leave)")
    st.pyplot(fig)

    # --- Department-wise Attrition ---
with col2:
    st.subheader("Attrition by Department")
    dept_attrition = df.groupby('department')['attrition'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5,3))
    sns.barplot(x='department', y='attrition', data=dept_attrition, ax=ax)
    ax.set_ylabel("Attrition Rate")
    st.pyplot(fig)

    # --- Monthly Income vs Attrition ---
col3, col4 = st.columns(2)
with col3:
    st.subheader("Monthly Income vs Attrition")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.boxplot(x='attrition', y='monthlyincome', data=df, ax=ax)
    st.pyplot(fig)

# --- Attrition vs Overtime ---
with col4:
    st.subheader("Attrition vs Overtime")
    fig, ax = plt.subplots(figsize=(5,3))
    sns.countplot(x='overtime', hue='attrition', data=df, ax=ax)
    st.pyplot(fig)

# --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# --------------------------
# TAB 2: Feature Importance
# --------------------------
with tabs[1]:
    st.header("Top Features Influencing Attrition")
    importances = rf_model.feature_importances_
    features = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=features[:15], y=features.index[:15], ax=ax, palette="viridis")
    ax.set_title("Top 15 Important Features")
    st.pyplot(fig)

# --------------------------
# TAB 3: Prediction
# --------------------------
with tabs[2]:
    st.header("Predict Employee Attrition")

    # Input Form
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.slider("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
        distance = st.slider("Distance From Home (km)", min_value=0, max_value=100, value=10, step=1)
        jobsatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
    with col2:
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        jobrole = st.selectbox("Job Role", df['jobrole'].unique())

    overtime = 1 if overtime == "Yes" else 0

    if st.button("Predict Attrition"):
        # Create a DataFrame with default 0s
        user_data = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)
        # Fill only the relevant fields
        if 'age' in user_data.columns: user_data['age'] = age
        if 'monthlyincome' in user_data.columns: user_data['monthlyincome'] = monthly_income
        if 'distancefromhome' in user_data.columns: user_data['distancefromhome'] = distance
        if 'jobsatisfaction' in user_data.columns: user_data['jobsatisfaction'] = jobsatisfaction
        if 'overtime' in user_data.columns: user_data['overtime'] = overtime
        if 'joblevel' in user_data.columns: user_data['joblevel'] = job_level
        if 'yearsatcompany' in user_data.columns: user_data['yearsatcompany'] = years_at_company
        if 'jobrole' in user_data.columns: user_data['jobrole'] = jobrole

        

        # Predict
        prediction = rf_model.predict(user_data)[0]
        proba = rf_model.predict_proba(user_data)[0]

        #extract probability for the predicted class
        predicted_prob = proba[prediction]

        #show results
        if prediction == 1:
            st.error(f"Prediction: **Leave** (Probability: {predicted_prob:.2f})")
        else:
            st.success(f"Prediction: **Stay** (Probability: {predicted_prob:.2f})")
            