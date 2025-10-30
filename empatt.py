# --------------------------
# Load Data and Model
# --------------------------
import json
from pathlib import Path
from joblib import load
import pandas as pd
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import base64
# --------------------------
# Streamlit App Setup
# --------------------------

def add_bg_from_local(image_file):
    """
    Function to load and set background image from local file
    """
    if image_file:
        with open(image_file, "rb") as f:
            encoded_string = base64.b64encode(f.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
add_bg_from_local('employee-attrition_1-20230105-082344.png')  # Replace with your image file path
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Analysis and Prediction Dashboard")


BASE_DIR = Path(__file__).parent

@st.cache_data
def load_raw_data():
    # Use your final cleaned csv for EDA if you want encoded EDA; OR the original raw csv for human-friendly plots.
    # If you still want the earlier raw (human-friendly) EDA, point to your original CSV (with strings).
    # Example tries final cleaned first, falls back to a raw one if present:
    for p in ["cleaned_employee_attrition_top_features.csv"]:
        fp = BASE_DIR / p
        if fp.exists():
            return pd.read_csv(fp)
    st.warning("Couldn't find a cleaned CSV. EDA will be limited.")
    return pd.DataFrame()

raw_df = load_raw_data()

# Load model + columns + threshold
rf_model = load(BASE_DIR /"random_forest_model (1).joblib")
with open(BASE_DIR / "feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

THRESHOLD = 0.35  # default

# Try to load threshold from metadata if available
meta_path = BASE_DIR /"metadata.json"
if meta_path.exists():
    with open(meta_path, "r") as f:
        meta = json.load(f)
        THRESHOLD = float(meta.get("threshold", THRESHOLD))



# --------------------------
# Tabs
# --------------------------


tabs = st.tabs(["EDA & Insights", "Feature Importance", "Prediction"])
        
with tabs[0]:
    st.header("Exploratory Data Analysis (EDA)")

    if raw_df.empty:
        st.info("No EDA dataset found.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attrition Count")
            fig, ax = plt.subplots(figsize=(6,3))
            sns.countplot(x='attrition', data=raw_df, ax=ax)
            ax.set_title("Attrition (No/Yes or 0/1)")
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with col2:
            if 'department' in raw_df.columns and 'attrition' in raw_df.columns:
                st.subheader("Attrition by Department")
                # Works for either 0/1 or Yes/No (coerce)
                tmp = raw_df.copy()
                if tmp['attrition'].dtype == object:
                    tmp['attrition'] = tmp['attrition'].map({'Yes':1,'No':0})
                dept_attr = tmp.groupby('department')['attrition'].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6,3))
                sns.barplot(x='department', y='attrition', data=dept_attr, ax=ax)
                ax.set_ylabel("Attrition Rate")
                ax.tick_params(axis='x', labelrotation=20)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                
        col3, col4 = st.columns(2)
        with col3:
            if 'monthlyincome' in raw_df.columns and 'attrition' in raw_df.columns:
                st.subheader("Monthly Income vs Attrition")
                fig, ax = plt.subplots(figsize=(6,3))
                sns.boxplot(x='attrition', y='monthlyincome', data=raw_df, ax=ax)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
        with col4:
            if 'overtime' in raw_df.columns and 'attrition' in raw_df.columns:
                st.subheader("Attrition vs Overtime")
                fig, ax = plt.subplots(figsize=(6,3))
                sns.countplot(x='overtime', hue='attrition', data=raw_df, ax=ax)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
        # Correlation on numeric-only columns if present
        st.subheader("Correlation Heatmap")
        corr = raw_df.corr(numeric_only=True)
        mask = corr.abs() > 0.3  # Show only meaningful correlations
        plt.figure(figsize=(10,6))
        sns.heatmap(corr[mask], annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        st.pyplot(plt)

with tabs[1]:
    st.header("Top Features Influencing Attrition")
    importances = rf_model.feature_importances_
    features = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=features.head(15), y=features.head(15).index, ax=ax, palette="viridis")
    ax.set_title("Top 15 Important Features")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# --------------------------
def get_category_options(prefix: str):
    """Return sorted list of category names for a one-hot group from feature_cols.
    E.g., prefix='jobrole_' -> ['Sales Executive', 'Research Scientist', ...]
    """
    pref = prefix.lower()
    opts = []
    for col in feature_cols:
        if col.lower().startswith(pref):
            # category is the part after the first underscore
            opts.append(col[len(prefix):])
    return sorted(set(opts))

def build_empty_row():
    return pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)

def set_one_hot(row: pd.DataFrame, prefix: str, chosen: str):
    """Set exactly one dummy to 1 given a chosen category name."""
    colname = prefix + chosen
    if colname in row.columns:
        row.at[0, colname] = 1
    else:
        # try exact-match fallback across cases/spaces
        for c in row.columns:
            if c.lower() == colname.lower():
                row.at[0, c] = 1
                break

with tabs[2]:
    st.header("Predict Employee Attrition")

    # Build category options by inspecting saved feature columns
    # Adjust prefixes to EXACTLY match how your dummies were created in Colab (case & spaces matter!)
    dept_opts  = get_category_options("department_")
    role_opts  = get_category_options("jobrole_")
    bt_opts    = get_category_options("businesstravel_")
    ms_opts    = get_category_options("maritalstatus_")
    ef_opts    = get_category_options("educationfield_")
    tc_opts    = get_category_options("tenurecategory_")
    gender_opts= get_category_options("gender_")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.slider("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
        distance = st.slider("Distance From Home (km)", min_value=0, max_value=100, value=10, step=1)
        jobsatisfaction = st.selectbox("Job Satisfaction (1–4)", [1,2,3,4], index=2)
        job_level = st.number_input("Job Level (1–5)", min_value=1, max_value=5, value=2)

    with col2:
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)

        # Friendly categorical selectors sourced from feature_cols
        # Show something sensible even if list is empty
        department = st.selectbox("Department", dept_opts or ["Research & Development","Sales","Human Resources"])
        jobrole    = st.selectbox("Job Role", role_opts or ["Research Scientist","Sales Executive","Laboratory Technician"])
        btravel    = st.selectbox("Business Travel", bt_opts or ["Travel_Rarely","Travel_Frequently","Non-Travel"])
        mstatus    = st.selectbox("Marital Status", ms_opts or ["Single","Married","Divorced"])
        edfield    = st.selectbox("Education Field", ef_opts or ["Life Sciences","Medical","Marketing","Technical Degree","Other"])
        tencat     = st.selectbox("Tenure Category", tc_opts or ["0-2 yrs","3-5 yrs","6-9 yrs","10+ yrs"])
        gender     = st.selectbox("Gender", gender_opts or ["Male","Female"])

    overtime_bin = 1 if overtime == "Yes" else 0

    if st.button("Predict Attrition"):
        user_row = build_empty_row()

        # numeric fields
        for k, v in [
            ("age", age),
            ("monthlyincome", monthly_income),
            ("distancefromhome", distance),
            ("jobsatisfaction", jobsatisfaction),
            ("overtime", overtime_bin),
            ("joblevel", job_level),
            ("yearsatcompany", years_at_company),
        ]:
            if k in user_row.columns:
                user_row.at[0, k] = v

        # one-hot fields (set exactly one per group)
        set_one_hot(user_row, "department_", department)
        set_one_hot(user_row, "jobrole_", jobrole)
        set_one_hot(user_row, "businesstravel_", btravel)
        set_one_hot(user_row, "maritalstatus_", mstatus)
        set_one_hot(user_row, "educationfield_", edfield)
        set_one_hot(user_row, "tenurecategory_", tencat)
        set_one_hot(user_row, "gender_", gender)

        # Predict with tuned threshold
        proba = rf_model.predict_proba(user_row)[0, 1]
        pred = int(proba >= THRESHOLD)

        if pred == 1:
            st.error(f"Prediction: **Leave** (Probability: {proba:.2f})  •  threshold={THRESHOLD:.2f}")
        else:
            st.success(f"Prediction: **Stay** (Probability: {proba:.2f})  •  threshold={THRESHOLD:.2f}")
