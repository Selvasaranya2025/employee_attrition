# 🧠 Employee Attrition Analysis & Prediction

### 👩‍💻 Author
**Selva Saranya S – DS‑S‑WD‑T‑B56**  
Master Data Science Program – IITM Incubation (GUVI) in collaboration with HCL  
[🌐 GitHub Profile](https://github.com/Selvasaranya2025/)  

---

## 📌 Project Overview
Predict employee attrition using Machine Learning and visualize HR insights through a Streamlit dashboard.  
This project aims to identify high‑risk employees and assist HR in reducing turnover rates.

---

## 🎯 Objectives
- Analyze employee data to identify key factors leading to attrition.  
- Build a predictive model to forecast employee exit risk.  
- Visualize and communicate results with an interactive dashboard.

---

## 🧩 Dataset
- **Source:** IBM HR Analytics Employee Attrition Dataset  
- **Rows:** 1470  
- **Target Variable:** `Attrition` (1 = Leave, 0 = Stay)  
- **Key Features:** Age, MonthlyIncome, JobRole, Department, OverTime, WorkLifeBalance, JobSatisfaction.

---

## ⚙️ Tech Stack
Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit‑Learn | Streamlit

---

## 🔍 Workflow
1. **Data Cleaning:** Removed duplicates, handled nulls, outliers capped using IQR.  
2. **Feature Engineering:** Created `EngagementScore`, `PromotionGap`, `OvertimeStress`, and `TenureCategory`.  
3. **EDA:** Visualized attrition rate by department, overtime, and job satisfaction.  
4. **Model Building:** Trained Random Forest Classifier, optimized threshold for recall.  
5. **Evaluation:** Achieved 81% accuracy, AUC = 0.82.  
6. **Deployment:** Streamlit dashboard with 3 functional tabs.

---

## 📊 Results

| Metric | Value |
|---------|-------|
| Accuracy | 81 % |
| Precision | 0.38 |
| Recall | 0.70 |
| F1‑Score | 0.49 |
| ROC‑AUC | 0.82 |

---

## 🖥️ Streamlit Dashboard
- **EDA Tab:** Explore attrition patterns and department‑wise trends.  
- **Feature Importance Tab:** Visualize top predictors.  
- **Prediction Tab:** HR can input employee details to get attrition probability.  

---

## 📦 Repository Structure
```
employee_attrition/
├── cleaned_employee_attrition_final.csv
├── empatt.py                # Streamlit app
├── employeeatt2.ipynb       # Model training notebook
├── random_forest_model.joblib
├── feature_columns.pkl
├── README.md
└── Employee_Attrition_Capstone_Presentation.pptx
```

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run empatt.py
```

---

## 📈 Business Impact
- Data‑driven HR policy development.  
- Reduced employee turnover by identifying high‑risk profiles.  
- Enhanced employee satisfaction tracking through analytics.

---

## 🔮 Future Scope
- Deploy on AWS / Streamlit Cloud.  
- Integrate Power BI dashboards.  
- Expand model with real‑time HR feedback data.

---
⭐ **Developed by:** [Selva Saranya S](https://github.com/Selvasaranya2025/)

