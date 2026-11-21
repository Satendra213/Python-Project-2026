# Retail Sales Predictive Analysis & Exploratory Data Analysis

**Course:** INT375 — Data Science Toolbox: Python Programming  
**Institution:** Lovely Professional University, Punjab  
**Discipline:** School of Computer Science and Engineering  
**Student Name:** Satendra  
**Registration Number:** 12320135  
**Course Faculty / Guide:** Vikas Mangotra (UID: 31488)  

---

## 📌 Project Overview
This project presents an end-to-end data science study analyzing retail sales performance across multiple product categories and regions in India using `retail_sales_eda_2000_rows.csv` (2,000 transactions, 16 attributes). The analysis strictly follows the syllabus of **INT375** covering:
- **Unit I & II:** Python fundamentals, NumPy array operations, Pandas Series/DataFrames, and Missing Data Handling.
- **Unit III & IV:** Data Visualization with Matplotlib/Seaborn, Exploratory Data Analysis (EDA), Covariance, Correlation, and Outlier Detection (IQR Method).
- **Unit V:** Statistical Analysis & Hypothesis Testing (Shapiro-Wilk test, One-sample t-test, Two-sample independent t-test for A/B testing, Chi-Square test of independence, VIF).
- **Unit VI:** Machine Learning under the CRISP-DM framework (Linear Regression for Sales forecasting and Logistic Regression for High-Profit order classification).

---

## 🎯 5 Core Problem Statements (Objectives)
1. **Data Cleaning & Missing Value Imputation (Unit II):** Audit missing records across `Payment_Method`, `Customer_Age`, `Customer_Rating`, and `Delivery_Days`, and apply mode/median imputation to ensure complete data integrity.
2. **Outlier Detection & Remediation (Unit IV):** Identify extreme financial values in `Unit_Price_INR`, `Sales_INR`, and `Profit_INR` using the $1.5 \times \text{IQR}$ rule and apply robust Winsorization (capping) without deleting observations.
3. **Exploratory Data Analysis & Visualization (Units III & IV):** Assess sales performance across product categories and regions using bar plots, line plots, scatter plots, box plots, pie charts, and correlation heatmaps.
4. **Statistical Hypothesis Testing (Unit V):** Test empirical assumptions:
   - *Normality:* Shapiro-Wilk test on `Profit_INR`.
   - *One-sample t-test:* Evaluating mean `Customer_Rating` against benchmark 4.0.
   - *Two-sample t-test (A/B testing):* Comparing revenue between Online and In-store channels.
   - *Chi-Square test:* Assessing independence between `Category` and order `Returned` status.
   - *Multicollinearity:* Computing Variance Inflation Factor (VIF).
5. **Supervised Machine Learning (Unit VI):** 
   - Fit a **Linear Regression** model predicting `Sales_INR` from `Units_Sold`, `Unit_Price_INR`, and `Discount_Percent` ($R^2 \approx 0.81$).
   - Fit a **Logistic Regression** classification model to predict `High_Profit` orders (Accuracy $\approx 83.75\%$, ROC-AUC $\approx 0.94$).

---

## 📂 Project Repository Structure
```
├── retail_sales_eda_2000_rows.csv     # Primary dataset (2,000 rows × 16 columns)
├── retail_sales_project.py            # Complete Python analysis script (runnable in IDLE)
├── Retail_Sales_Project_Report.docx   # Comprehensive University Project Report (LPU format)
├── report_charts/                     # High-resolution publication charts
│   ├── fig_missing.png                # Missing values count before and after
│   ├── fig_pie.png                    # Target class distribution pie chart
│   ├── fig_univariate.png             # Sales histogram (KDE) and category countplot
│   ├── fig_outliers.png               # Boxplots before and after IQR capping
│   ├── fig_correlation.png            # Full numerical correlation heatmap
│   ├── fig_corr_bar.png               # Correlation with target variable
│   ├── fig_bivariate.png              # Regional sales and category profit boxplots
│   └── fig_ml.png                     # Linear Regression scatter, Confusion Matrix, ROC
├── 01_missing_values.png              # Standalone chart: Missing values
├── 02_outliers_iqr.png                # Standalone chart: Outlier analysis
├── 03_eda_visualizations.png          # Standalone chart: EDA distributions
├── 04_correlation_heatmap.png         # Standalone chart: Correlation matrix
├── 05_ml_models.png                   # Standalone chart: ML diagnostics
├── .gitignore                         # Git ignore rules for clean repository
└── README.md                          # Project documentation
```

---

## 🚀 How to Run the Code
### Option 1: In Python IDLE
1. Open **Python IDLE**.
2. Go to **File → Open...** and select `retail_sales_project.py`.
3. Press **F5** (or **Run → Run Module**).
4. The numerical findings and statistical test results will display in the IDLE shell, and interactive visualization windows will appear.

### Option 2: From Terminal / Command Prompt
```bash
python retail_sales_project.py
```

---

## 📊 Summary of Results
- **Missing Values:** Successfully resolved 190 missing values (0 nulls remaining).
- **Outlier Capping:** Applied $1.5 \times \text{IQR}$ bounds on `Sales_INR` and `Profit_INR` without row loss.
- **Normality Test:** Shapiro-Wilk test yielded $p < 0.05$ (Reject $H_0$; distribution is positively skewed).
- **Channel Comparison (A/B Test):** Two-sample t-test between Online and In-store channels resulted in $p = 0.4296$ (Fail to reject $H_0$; no significant sales divergence).
- **Linear Regression:** Achieved $R^2 = 0.8124$ with significant positive coefficients on `Units_Sold` and `Unit_Price_INR`.
- **Classification Model:** Logistic Regression achieved $83.75\%$ classification accuracy and a $0.9364$ ROC-AUC score for predicting high-profit transactions.
