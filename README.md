# employee_attrition
1. Project Goal
The goal of this project is to analyze employee data and predict which employees are likely to leave (attrition).

This helps HR teams reduce attrition by improving work conditions, salaries, etc.

2. Dataset Overview
You are using an Employee Attrition dataset with columns like:

Demographics: Age, Gender, MaritalStatus

Work-related: Department, JobRole, YearsAtCompany, OverTime

Performance-related: JobSatisfaction, PerformanceRating

Compensation: MonthlyIncome, PercentSalaryHike

Target Variable: Attrition (Yes/No → 1/0)

3. Steps You Followed
A. Data Cleaning
Removed useless columns (EmployeeCount, StandardHours, etc.).

Converted text labels to numeric (e.g., Attrition: Yes=1, No=0).

Standardized column names (e.g., MonthlyIncome → monthlyincome).

B. Exploratory Data Analysis (EDA)
Attrition Rate: Checked how many employees left.

Charts:

Attrition vs JobRole – Which roles have high turnover.

Monthly Income vs Attrition – Lower salary = higher attrition.

Overtime vs Attrition – Overtime workers leave more.

Heatmap – To see correlation (e.g., joblevel and monthlyincome are highly correlated).

C. Feature Engineering
You created new features:

tenure_category – Grouped employees by years at company.

performance_score – Combined performance rating and salary hike.

engagement_score – Average of satisfaction metrics.

overtime_stress – Overtime × Years at company.

income_per_year – Salary fairness relative to total experience.

D. Encoding
Label Encoding: For columns like Gender, Overtime.

One-Hot Encoding: For categorical columns like Department, JobRole.

E. Model Building
Logistic Regression: Your baseline model (accuracy ~71%, recall ~63%).

Random Forest Classifier: Performed better and gave feature importance.

4. Streamlit Dashboard
Displays EDA visualizations (attrition trends, salary patterns, heatmap).

Shows important features (like overtime, monthly income).

Provides a prediction form to check if a given employee might leave.

5. Key Insights
Overtime is the strongest predictor of attrition.

Low salary and low job satisfaction increase attrition risk.

Sales Executive and Laboratory Technician roles have the highest turnover.

Employees with fewer years at the company are more likely to leave.

