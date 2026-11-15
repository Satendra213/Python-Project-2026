# ==============================================================================
# Python Project: Retail Sales Predictive Analysis
# Dataset: retail_sales_eda_2000_rows.csv
#
# 5 PROBLEM STATEMENTS (OBJECTIVES):
# 1. Inspect and clean missing data using mode and median imputation.
# 2. Detect and handle outliers using the IQR (Interquartile Range) method.
# 3. Perform Exploratory Data Analysis using Bar, Line, Scatter, Box, Pie, and Heatmap.
# 4. Perform Hypothesis Testing: Shapiro-Wilk, One-Sample t-test, Two-Sample t-test, and Chi-Square test.
# 5. Build Machine Learning Models: Linear Regression (for Sales) and Classification (for High Profit).
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------------------------------
# STEP 1: LOAD DATA AND INITIAL INSPECTION (info & describe)
# ------------------------------------------------------------------------------
try:
    df = pd.read_csv('05_retail_sales_eda_2000_rows.csv')
except FileNotFoundError:
    df = pd.read_csv('retail_sales_eda_2000_rows.csv')

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nNumerical Summary (describe):")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include='object'))


# ------------------------------------------------------------------------------
# STEP 2: HANDLING MISSING VALUES
# ------------------------------------------------------------------------------
print("\n--- Missing Values in Each Column ---")
print(df.isnull().sum())

# Missing values before cleaning plot
missing = df.isnull().sum()
missing = missing[missing > 0]
missing.plot(kind='bar', color='coral')
plt.title("Missing Values Count")
plt.xlabel("Columns")
plt.ylabel("Missing Count")
plt.show()

# Imputation (Median for numerical, Mode for categorical)
df['Customer_Age'] = df['Customer_Age'].fillna(df['Customer_Age'].median())
df['Customer_Rating'] = df['Customer_Rating'].fillna(df['Customer_Rating'].median())
df['Delivery_Days'] = df['Delivery_Days'].fillna(df['Delivery_Days'].median())
df['Payment_Method'] = df['Payment_Method'].fillna(df['Payment_Method'].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------------------------
# STEP 3: OUTLIER DETECTION (IQR METHOD)
# ------------------------------------------------------------------------------
print("\n--- Outlier Detection using IQR ---")
# Box plot to visualize outliers before treatment
df[['Unit_Price_INR', 'Sales_INR', 'Profit_INR']].plot(kind='box')
plt.title("Box Plot for Outlier Detection (Before Capping)")
plt.ylabel("INR")
plt.show()

# IQR Calculation for Sales_INR
Q1_sales = df['Sales_INR'].quantile(0.25)
Q3_sales = df['Sales_INR'].quantile(0.75)
IQR_sales = Q3_sales - Q1_sales
lower_sales = Q1_sales - 1.5 * IQR_sales
upper_sales = Q3_sales + 1.5 * IQR_sales
outliers_sales = df[(df['Sales_INR'] < lower_sales) | (df['Sales_INR'] > upper_sales)]
print(f"Sales_INR: Q1={Q1_sales:.2f}, Q3={Q3_sales:.2f}, IQR={IQR_sales:.2f}")
print(f"Lower Bound={lower_sales:.2f}, Upper Bound={upper_sales:.2f}")
print(f"Number of Outliers in Sales_INR: {len(outliers_sales)}")

# IQR Calculation for Profit_INR
Q1_profit = df['Profit_INR'].quantile(0.25)
Q3_profit = df['Profit_INR'].quantile(0.75)
IQR_profit = Q3_profit - Q1_profit
lower_profit = Q1_profit - 1.5 * IQR_profit
upper_profit = Q3_profit + 1.5 * IQR_profit
outliers_profit = df[(df['Profit_INR'] < lower_profit) | (df['Profit_INR'] > upper_profit)]
print(f"Profit_INR: Q1={Q1_profit:.2f}, Q3={Q3_profit:.2f}, IQR={IQR_profit:.2f}")
print(f"Number of Outliers in Profit_INR: {len(outliers_profit)}")

# Capping outliers (Winsorization)
df['Sales_INR_Cleaned'] = df['Sales_INR'].clip(lower_sales, upper_sales)
df['Profit_INR_Cleaned'] = df['Profit_INR'].clip(lower_profit, upper_profit)

# Box plot after capping
df[['Sales_INR_Cleaned', 'Profit_INR_Cleaned']].plot(kind='box')
plt.title("Box Plot After Outlier Capping")
plt.ylabel("INR")
plt.show()


# ------------------------------------------------------------------------------
# STEP 4: DATA VISUALIZATION (CHARTS AND GRAPHS)
# ------------------------------------------------------------------------------
# 1. Bar plot: Sales by Category
category_sales = df.groupby('Category')['Sales_INR'].sum()
category_sales.plot(kind='bar', color='skyblue')
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales (INR)")
plt.show()

# 2. Line plot: Monthly Sales Trend
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
monthly_sales = df.set_index('Order_Date').resample('ME')['Sales_INR'].sum()
monthly_sales.plot(kind='line', color='green', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales (INR)")
plt.show()

# 3. Scatter plot: Units Sold vs Sales
plt.scatter(df['Units_Sold'], df['Sales_INR'], color='purple', alpha=0.5)
plt.title("Scatter Plot: Units Sold vs Sales (INR)")
plt.xlabel("Units Sold")
plt.ylabel("Sales (INR)")
plt.show()

# 4. Pie chart: Sales Channel Distribution
channel_counts = df['Sales_Channel'].value_counts()
plt.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Sales Channel Distribution")
plt.show()

# 5. Correlation Heatmap
numeric_df = df[['Units_Sold', 'Unit_Price_INR', 'Discount_Percent', 'Sales_INR', 'Profit_INR', 'Customer_Age', 'Customer_Rating', 'Delivery_Days']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ------------------------------------------------------------------------------
# STEP 5: STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# ------------------------------------------------------------------------------
alpha = 0.05

# 1. Shapiro-Wilk Test (Check if Profit_INR is normally distributed)
stat_shapiro, p_val_shapiro = stats.shapiro(df['Profit_INR'][:500])
print("\n--- 1. Shapiro-Wilk Test (Normality Test) ---")
print(f"Statistic: {stat_shapiro:.4f}, P-value: {p_val_shapiro:.4e}")
if p_val_shapiro > alpha:
    print("Conclusion: Fail to reject H0 (Data is normally distributed)")
else:
    print("Conclusion: Reject H0 (Data is not normally distributed)")

# 2. One-Sample t-test (Testing if mean Customer Rating is equal to 4.0)
pop_mean_rating = 4.0
t_stat_one, p_val_one = stats.ttest_1samp(df['Customer_Rating'], pop_mean_rating)
print("\n--- 2. One-Sample t-test (Customer Rating vs 4.0) ---")
print(f"Sample Mean: {df['Customer_Rating'].mean():.2f}")
print(f"T-statistic: {t_stat_one:.4f}, P-value: {p_val_one:.4f}")
if p_val_one < alpha:
    print("Conclusion: Reject H0 (Mean rating is significantly different from 4.0)")
else:
    print("Conclusion: Fail to reject H0 (Mean rating is equal to 4.0)")

# 3. Two-Sample Independent t-test (A/B Testing: Online vs In-store Sales)
online_sales = df[df['Sales_Channel'] == 'Online']['Sales_INR']
instore_sales = df[df['Sales_Channel'] == 'In-store']['Sales_INR']
t_stat_two, p_val_two = stats.ttest_ind(online_sales, instore_sales, equal_var=False)
print("\n--- 3. Two-Sample t-test (Online vs In-store Sales) ---")
print(f"Online Mean: {online_sales.mean():.2f}, In-store Mean: {instore_sales.mean():.2f}")
print(f"T-statistic: {t_stat_two:.4f}, P-value: {p_val_two:.4f}")
if p_val_two < alpha:
    print("Conclusion: Reject H0 (Significant difference between Online and In-store Sales)")
else:
    print("Conclusion: Fail to reject H0 (No significant difference between Online and In-store Sales)")

# 4. Chi-Square Test of Independence (Category vs Returned)
contingency = pd.crosstab(df['Category'], df['Returned'])
chi2_stat, p_val_chi2, dof, expected = chi2_contingency(contingency)
print("\n--- 4. Chi-Square Test of Independence (Category vs Returned) ---")
print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_val_chi2:.4f}")
if p_val_chi2 < alpha:
    print("Conclusion: Reject H0 (Significant relationship between Category and Returned)")
else:
    print("Conclusion: Fail to reject H0 (Category and Returned are independent)")


# ------------------------------------------------------------------------------
# STEP 6: MACHINE LEARNING (CRISP-DM FRAMEWORK)
# ------------------------------------------------------------------------------
# A. Linear Regression (Predict Sales_INR)
print("\n--- Linear Regression Model ---")
X_reg = df[['Units_Sold', 'Unit_Price_INR', 'Discount_Percent']]
y_reg = df['Sales_INR_Cleaned']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

model_reg = LinearRegression()
model_reg.fit(X_train_r, y_train_r)
y_pred_r = model_reg.predict(X_test_r)

print("Mean Squared Error (MSE):", mean_squared_error(y_test_r, y_pred_r))
print(f"R^2 Score: {r2_score(y_test_r, y_pred_r):.4f}")
print(f"Intercept: {model_reg.intercept_:.2f}")
print(f"Coefficients: {model_reg.coef_}")

# Scatter plot: Actual vs Predicted Sales
plt.scatter(y_test_r, y_pred_r, color='blue', alpha=0.5)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.title("Linear Regression: Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.show()

# B. Classification (Logistic Regression: Predict High Profit)
print("\n--- Classification Model (Logistic Regression) ---")
# 1 if profit is greater than median profit, else 0
df['High_Profit'] = (df['Profit_INR'] > df['Profit_INR'].median()).astype(int)

X_clf = pd.get_dummies(df[['Units_Sold', 'Unit_Price_INR', 'Discount_Percent', 'Customer_Age', 'Delivery_Days', 'Category']], drop_first=True)
y_clf = df['High_Profit']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

model_clf = LogisticRegression(solver='liblinear')
model_clf.fit(X_train_c, y_train_c)
y_pred_c = model_clf.predict(X_test_c)

print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c) * 100:.2f}%")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_c, y_pred_c)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (High Profit Classification)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
